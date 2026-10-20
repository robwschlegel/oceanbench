# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import dask.array
import numpy
import pandas
import pytest
import xarray

from oceanbench.core.dataset_utils import Variable
from oceanbench.core.references import ibi36
from oceanbench.datasets.reference import ibi36_reference


DEPTHS = numpy.array([0.5, 50.0, 100.0])
LATITUDES = numpy.array([40.0, 40.0277, 40.0554])
LONGITUDES = numpy.array([-10.0, -9.9723, -9.9446])
TEMPERATURE = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
SALINITY = Variable.SEA_WATER_SALINITY.key()
EASTWARD_CURRENT = Variable.EASTWARD_SEA_WATER_VELOCITY.key()
NORTHWARD_CURRENT = Variable.NORTHWARD_SEA_WATER_VELOCITY.key()
HEIGHT = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()

_DATASET_IDS = {
    "thetao": "cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m",
    "so": "cmems_mod_ibi_phy-sal_my_0.027deg_P1D-m",
    "uo": "cmems_mod_ibi_phy-cur_my_0.027deg_P1D-m",
    "vo": "cmems_mod_ibi_phy-cur_my_0.027deg_P1D-m",
    "zos": "cmems_mod_ibi_phy-ssh_my_0.027deg_P1D-m",
}


def _variable_dataset(name, times):
    dims = ("time", "latitude", "longitude") if name == "zos" else ("time", "depth", "latitude", "longitude")
    shape = (len(times), 3, 3) if name == "zos" else (len(times), 3, 3, 3)
    value = {"thetao": 18.0, "so": 35.0, "uo": 0.25, "vo": -0.125, "zos": 0.1}[name]
    standard_name = {
        "thetao": "sea_water_potential_temperature",
        "so": "sea_water_practical_salinity",
        "uo": "eastward_sea_water_velocity",
        "vo": "northward_sea_water_velocity",
        "zos": "sea_surface_height_above_geoid",
    }[name]
    units = {"thetao": "degrees_C", "so": "1e-3", "uo": "m s-1", "vo": "m/s", "zos": "m"}[name]
    coords = {
        "time": times,
        "latitude": LATITUDES,
        "longitude": LONGITUDES,
    }
    if name != "zos":
        coords["depth"] = DEPTHS
    values = dask.array.from_array(numpy.full(shape, value), chunks=tuple(1 for _ in shape))
    return xarray.Dataset(
        {name: (dims, values, {"standard_name": standard_name, "units": units})},
        coords=coords,
    )


def _mask_dataset():
    values = numpy.ones((3, 3, 3))
    values[:, 0, 0] = 0
    values[2, 1, 1] = 0
    return xarray.Dataset(
        {"mask": (("depth", "latitude", "longitude"), dask.array.from_array(values, chunks=(1, 3, 3)))},
        coords={"depth": DEPTHS, "latitude": LATITUDES, "longitude": LONGITUDES},
    )


def _patch_open_dataset(monkeypatch, *, invalid_mask=False, fail_on=None):
    calls, opened, closed = [], [], []

    def apply_bounds(dataset, parameters):
        for coordinate in ("longitude", "latitude", "depth"):
            if coordinate not in dataset.coords:
                continue
            values = dataset[coordinate].values
            minimum = parameters.get(f"minimum_{coordinate}")
            maximum = parameters.get(f"maximum_{coordinate}")
            if minimum is not None:
                values = values[values >= minimum]
            if maximum is not None:
                values = values[values <= maximum]
            if values.size:
                dataset = dataset.sel({coordinate: values})
        return dataset

    def fake_open_dataset(**parameters):
        calls.append(parameters)
        dataset_id = parameters["dataset_id"]
        if dataset_id == fail_on:
            raise RuntimeError("simulated remote-open failure")
        if parameters["dataset_id"] == "cmems_mod_ibi_phy_my_0.027deg-3D_static":
            dataset = _mask_dataset()
            if invalid_mask:
                dataset = dataset.assign_coords(latitude=dataset.latitude + 1)
        else:
            times = pandas.date_range(
                parameters["start_datetime"],
                parameters["end_datetime"],
                freq="D",
            ).values
            dataset = xarray.merge([_variable_dataset(name, times) for name in parameters["variables"]])
        dataset = apply_bounds(dataset, parameters)

        def close():
            closed.append(dataset_id)

        dataset.set_close(close)
        opened.append(dataset_id)
        return dataset

    monkeypatch.setattr(ibi36.copernicusmarine, "open_dataset", fake_open_dataset)
    return calls, opened, closed


def test_public_reader_opens_versioned_sources_lazily_across_years(monkeypatch):
    calls, opened, closed = _patch_open_dataset(monkeypatch)
    with ibi36_reference("2023-12-31", "2024-01-01", "user", "password") as dataset:
        assert dataset.sizes == {"time": 2, "depth": 3, "latitude": 3, "longitude": 3}
        assert set(dataset.data_vars) == {TEMPERATURE, SALINITY, EASTWARD_CURRENT, NORTHWARD_CURRENT, HEIGHT}
        assert isinstance(dataset[TEMPERATURE].data, dask.array.Array)
        numpy.testing.assert_array_equal(
            dataset.time,
            numpy.array(["2023-12-31T00:00", "2024-01-01T00:00"], dtype="datetime64[ns]"),
        )
        for name, values in (("depth", DEPTHS), ("latitude", LATITUDES), ("longitude", LONGITUDES)):
            numpy.testing.assert_array_equal(dataset[name], values)
        assert dataset.attrs["ibi36_source_product"] == "IBI_MULTIYEAR_PHY_005_002"
        assert dataset.attrs["ibi36_dataset_version"] == "202511"
    assert len(opened) == 5
    assert sorted(closed) == sorted(opened)
    data_calls = [call for call in calls if call["dataset_id"] != "cmems_mod_ibi_phy_my_0.027deg-3D_static"]
    assert {call["dataset_id"] for call in data_calls} == set(_DATASET_IDS.values())
    assert len(data_calls) == 4
    current_call = next(call for call in data_calls if call["dataset_id"] == _DATASET_IDS["uo"])
    assert current_call["variables"] == ["uo", "vo"]
    assert {call["dataset_version"] for call in calls} == {"202511"}
    assert {call["start_datetime"] for call in data_calls} == {"2023-12-31T00:00:00"}
    assert {call["end_datetime"] for call in data_calls} == {"2024-01-01T00:00:00"}
    assert {call["username"] for call in calls} == {"user"}
    assert {call["password"] for call in calls} == {"password"}
    assert all(call["vertical_axis"] == "depth" for call in data_calls if call["variables"] != ["zos"])
    assert "vertical_axis" not in next(call for call in data_calls if call["variables"] == ["zos"])
    mask_call = next(call for call in calls if call["dataset_id"] == "cmems_mod_ibi_phy_my_0.027deg-3D_static")
    assert mask_call["dataset_part"] == "bathy"
    assert mask_call["variables"] == ["mask"]
    assert mask_call["vertical_axis"] == "depth"


def test_reader_keeps_unchunked_toolbox_sources_lazy(monkeypatch):
    _patch_open_dataset(monkeypatch)
    original_open = ibi36.copernicusmarine.open_dataset

    def unchunked_open(**parameters):
        source = original_open(**parameters)
        dataset = source.compute()
        dataset.set_close(source.close)
        return dataset

    monkeypatch.setattr(ibi36.copernicusmarine, "open_dataset", unchunked_open)
    with ibi36_reference("2023-12-31", "2023-12-31") as dataset:
        assert all(isinstance(variable.data, dask.array.Array) for variable in dataset.data_vars.values())


def test_reader_masks_land_and_below_bottom_without_rescaling_salinity(monkeypatch):
    _patch_open_dataset(monkeypatch)
    with ibi36_reference("2023-12-31", "2023-12-31") as dataset:
        for name in (TEMPERATURE, SALINITY, EASTWARD_CURRENT, NORTHWARD_CURRENT):
            values = dataset[name].compute()
            assert numpy.isnan(values.isel(latitude=0, longitude=0)).all()
            assert numpy.isnan(values.isel(depth=2, latitude=1, longitude=1)).all()
            assert numpy.isfinite(values.isel(depth=1, latitude=1, longitude=1)).all()
        height = dataset[HEIGHT].compute()
        assert height.dims == ("time", "latitude", "longitude")
        assert numpy.isnan(height.isel(latitude=0, longitude=0)).all()
        assert height.isel(time=0, latitude=1, longitude=1).item() == 0.1
        assert dataset[SALINITY].attrs["units"] == "1e-3"
        assert dataset[SALINITY].isel(time=0, depth=1, latitude=2, longitude=2).compute().item() == 35


def test_currents_keep_their_direction_sign_units_and_depths(monkeypatch):
    _patch_open_dataset(monkeypatch)
    with ibi36_reference("2023-12-31", "2023-12-31", minimum_depth=50) as dataset:
        for name, expected in ((EASTWARD_CURRENT, 0.25), (NORTHWARD_CURRENT, -0.125)):
            current = dataset[name]
            assert current.dims == ("time", "depth", "latitude", "longitude")
            assert current.attrs["standard_name"] == name
            assert current.attrs["units"] == "m/s"
            numpy.testing.assert_array_equal(current.depth, [50, 100])
            numpy.testing.assert_allclose(current.isel(latitude=2, longitude=2).compute(), expected)


def test_public_reader_subsets_data_and_mask_to_the_requested_region(monkeypatch):
    calls, _, _ = _patch_open_dataset(monkeypatch)
    with ibi36_reference(
        "2023-12-31",
        "2023-12-31",
        minimum_longitude=-10,
        maximum_longitude=-9.9723,
        minimum_latitude=40,
        maximum_latitude=40.0277,
        minimum_depth=50,
        maximum_depth=100,
    ) as dataset:
        assert dataset.sizes == {"time": 1, "depth": 2, "latitude": 2, "longitude": 2}
        numpy.testing.assert_array_equal(dataset.depth, [50, 100])

    for call in calls:
        assert call["minimum_longitude"] == -10
        assert call["maximum_longitude"] == -9.9723
        assert call["minimum_latitude"] == 40
        assert call["maximum_latitude"] == 40.0277
        assert call["coordinates_selection_method"] == "inside"
        if call["dataset_id"] in {_DATASET_IDS["thetao"], _DATASET_IDS["so"], _DATASET_IDS["uo"]}:
            assert call["minimum_depth"] == 50
            assert call["maximum_depth"] == 100
        elif call["dataset_id"] == _DATASET_IDS["zos"]:
            assert "minimum_depth" not in call
            assert "maximum_depth" not in call
        else:
            assert "minimum_depth" not in call
            assert call["maximum_depth"] == 100


@pytest.mark.parametrize("minimum_depth", [50, 100])
def test_surface_height_does_not_depend_on_selected_depths(monkeypatch, minimum_depth):
    _patch_open_dataset(monkeypatch)
    with (
        ibi36_reference("2023-12-31", "2023-12-31") as full_depth_reference,
        ibi36_reference("2023-12-31", "2023-12-31", minimum_depth=minimum_depth, maximum_depth=100) as deep_reference,
    ):
        xarray.testing.assert_equal(deep_reference[HEIGHT].compute(), full_depth_reference[HEIGHT].compute())
        assert deep_reference[HEIGHT].isel(time=0, latitude=1, longitude=1).compute().item() == 0.1
        assert numpy.isnan(deep_reference[TEMPERATURE].sel(depth=100).isel(latitude=1, longitude=1).compute()).all()


@pytest.mark.parametrize(
    "name, standard_name, units",
    [
        ("thetao", "sea_water_temperature", "degrees_C"),
        ("so", "sea_water_absolute_salinity", "kg kg-1"),
        ("uo", "northward_sea_water_velocity", "m/s"),
        ("vo", "northward_sea_water_velocity", "cm/s"),
        ("zos", "sea_surface_height_above_mean_sea_level", "m"),
    ],
)
def test_incompatible_source_metadata_is_rejected(monkeypatch, name, standard_name, units):
    _patch_open_dataset(monkeypatch)
    original_open = ibi36.copernicusmarine.open_dataset

    def incompatible_open(**parameters):
        dataset = original_open(**parameters)
        if name in parameters["variables"]:
            dataset[name].attrs.update(standard_name=standard_name, units=units)
        return dataset

    monkeypatch.setattr(ibi36.copernicusmarine, "open_dataset", incompatible_open)
    with pytest.raises(ValueError):
        ibi36_reference("2023-12-31", "2023-12-31")


def test_daily_source_requires_midnight_timestamps(monkeypatch):
    _patch_open_dataset(monkeypatch)
    original_open = ibi36.copernicusmarine.open_dataset

    def wrong_time(**parameters):
        dataset = original_open(**parameters)
        if parameters["dataset_id"] == _DATASET_IDS["thetao"]:
            dataset = dataset.assign_coords(time=[numpy.datetime64("2023-12-31T12:00")])
        return dataset

    monkeypatch.setattr(ibi36.copernicusmarine, "open_dataset", wrong_time)
    with pytest.raises(ValueError, match="timestamps"):
        ibi36_reference("2023-12-31", "2023-12-31")


def test_static_mask_grid_must_match_data(monkeypatch):
    _, opened, closed = _patch_open_dataset(monkeypatch, invalid_mask=True)
    with pytest.raises(ValueError, match="latitude"):
        ibi36_reference("2023-12-31", "2023-12-31")
    assert len(opened) == 5
    assert sorted(closed) == sorted(opened)


@pytest.mark.parametrize(
    "start, end, limits",
    [
        ("invalid", "2023-12-31", {}),
        ("2024-01-01", "2023-12-31", {}),
        ("2023-12-31", "2023-12-31", {"minimum_depth": -1}),
        (
            "2023-12-31",
            "2023-12-31",
            {"minimum_latitude": 42, "maximum_latitude": 41},
        ),
    ],
)
def test_invalid_dates_and_limits_are_rejected_before_opening(start, end, limits, monkeypatch):
    calls, _, _ = _patch_open_dataset(monkeypatch)
    with pytest.raises(ValueError):
        ibi36_reference(start, end, **limits)
    assert not calls


def test_all_open_datasets_close_if_opening_later_source_fails(monkeypatch):
    _, opened, closed = _patch_open_dataset(monkeypatch, fail_on=_DATASET_IDS["zos"])
    with pytest.raises(RuntimeError, match="simulated remote-open failure"):
        ibi36_reference("2023-12-31", "2023-12-31")
    assert len(opened) == 3
    assert sorted(closed) == sorted(opened)
