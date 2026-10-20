# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
import xarray
from dask.base import is_dask_collection
from dask.callbacks import Callback

from oceanbench.core.dataset_source import DatasetSource, get_dataset_source, with_dataset_source
from oceanbench.core.interpolate import interpolate_1_degree, interpolate_to_grid


def _dataset(values: numpy.ndarray) -> xarray.Dataset:
    return xarray.Dataset(
        {"thetao": (["time", "lat", "lon"], values, {"standard_name": "sea_water_potential_temperature"})},
        coords={
            "time": numpy.array(["2024-01-03", "2024-01-04"], dtype="datetime64[ns]"),
            "lat": xarray.DataArray([-1.0, 0.0, 1.0], dims=["lat"], attrs={"standard_name": "latitude"}),
            "lon": xarray.DataArray([10.0, 11.0, 12.0], dims=["lon"], attrs={"standard_name": "longitude"}),
        },
    )


def test_interpolate_1_degree_targets_half_degree_centres_inside_the_grid_and_interpolates_linearly() -> None:
    # thetao = 3 * latitude_index + longitude_index (+ 9 on the second day): a plane, so linear
    # interpolation at the half-degree centres is exact and can be written down by hand.
    interpolated = interpolate_1_degree(_dataset(numpy.arange(18, dtype=float).reshape(2, 3, 3)))

    assert interpolated["latitude"].values.tolist() == [-0.5, 0.5]
    assert interpolated["longitude"].values.tolist() == [10.5, 11.5]
    numpy.testing.assert_allclose(
        interpolated["sea_water_potential_temperature"].values,
        [[[2.0, 3.0], [5.0, 6.0]], [[11.0, 12.0], [14.0, 15.0]]],
    )


@pytest.mark.parametrize("name", ["glonet", "glo12", "glorys", "ibi36"])
def test_interpolate_1_degree_marks_every_dataset_source_as_one_degree(name: str) -> None:
    dataset = with_dataset_source(_dataset(numpy.zeros((2, 3, 3))), kind="reference", name=name)

    interpolated = interpolate_1_degree(dataset)

    assert get_dataset_source(interpolated) == DatasetSource(
        kind="reference",
        name=name,
        resolution="one_degree",
    )
    numpy.testing.assert_array_equal(interpolated.latitude, [-0.5, 0.5])
    numpy.testing.assert_array_equal(interpolated.longitude, [10.5, 11.5])


def test_interpolate_1_degree_preserves_missing_dataset_source() -> None:
    interpolated = interpolate_1_degree(_dataset(numpy.zeros((2, 3, 3))))

    assert get_dataset_source(interpolated) is None


def _fine_dataset() -> xarray.Dataset:
    latitude = 44.75 + numpy.arange(19) / 36
    longitude = -7.75 + numpy.arange(19) / 36
    return with_dataset_source(
        xarray.Dataset(
            {"temperature": (("latitude", "longitude"), 3 * latitude[:, None] + longitude[None, :])},
            coords={"latitude": latitude, "longitude": longitude},
        ),
        kind="reference",
        name="ibi36",
        resolution="thirty_sixth_degree",
        variant="copernicus",
    )


def _target_grid() -> xarray.Dataset:
    return xarray.Dataset(
        coords={
            "latitude": [44.91666666666667, 45.0, 45.08333333333333],
            "longitude": [-7.583333333333343, -7.5, -7.416666666666657],
        }
    )


def test_interpolate_to_grid_preserves_the_actual_target_coordinates_and_updates_provenance() -> None:
    target = _target_grid()

    interpolated = interpolate_to_grid(_fine_dataset(), target)

    numpy.testing.assert_array_equal(interpolated.latitude, target.latitude)
    numpy.testing.assert_array_equal(interpolated.longitude, target.longitude)
    numpy.testing.assert_allclose(
        interpolated.temperature,
        3 * target.latitude.values[:, None] + target.longitude.values[None, :],
    )
    assert get_dataset_source(interpolated) == DatasetSource(
        kind="reference",
        name="ibi36",
        resolution="twelfth_degree",
        variant="copernicus",
    )


def test_interpolate_to_grid_is_lazy_and_crops_before_rechunking(monkeypatch) -> None:
    dataset = _fine_dataset().chunk({"latitude": 3, "longitude": 3})
    rechunked_sizes = []
    original_chunk = xarray.Dataset.chunk

    def _record_chunk(dataset, *arguments, **keywords):
        rechunked_sizes.append(dict(dataset.sizes))
        return original_chunk(dataset, *arguments, **keywords)

    monkeypatch.setattr(xarray.Dataset, "chunk", _record_chunk)
    computed_tasks = []
    with Callback(pretask=lambda *arguments: computed_tasks.append(arguments[0])):
        interpolated = interpolate_to_grid(dataset, _target_grid())

    assert is_dask_collection(interpolated.temperature.data)
    assert not computed_tasks
    assert rechunked_sizes[0]["latitude"] < dataset.sizes["latitude"]
    assert rechunked_sizes[0]["longitude"] < dataset.sizes["longitude"]


@pytest.mark.parametrize("name", ["latitude", "longitude"])
def test_interpolate_to_grid_rejects_extrapolation(name: str) -> None:
    target = _target_grid().assign_coords({name: [100.0, 101.0]})

    with pytest.raises(ValueError, match=f"Target {name}.*extrapolation"):
        interpolate_to_grid(_fine_dataset(), target)


@pytest.mark.parametrize("values", [[], [numpy.nan, 45.0], [45.0, 45.0], [45.0, 45.1, 44.9]])
@pytest.mark.parametrize("is_target", [False, True])
def test_interpolate_to_grid_rejects_invalid_axes(values: list, is_target: bool) -> None:
    dataset = _fine_dataset()
    target = _target_grid()
    if is_target:
        target = target.drop_dims("latitude").assign_coords(latitude=values)
    else:
        dataset = dataset.drop_dims("latitude").assign_coords(latitude=values)

    with pytest.raises(ValueError, match="latitude"):
        interpolate_to_grid(dataset, target)


def test_interpolate_to_grid_rejects_curvilinear_target_coordinates() -> None:
    target = xarray.Dataset(coords={"latitude": (("y", "x"), [[45.0]]), "longitude": [0.0]})

    with pytest.raises(ValueError, match="1D latitude"):
        interpolate_to_grid(_fine_dataset(), target)


def test_interpolate_to_grid_requires_two_source_points_for_linear_interpolation() -> None:
    with pytest.raises(ValueError, match="two source latitude"):
        interpolate_to_grid(_fine_dataset().isel(latitude=[9]), _target_grid())


def test_interpolate_to_grid_accepts_descending_axes_and_cf_names() -> None:
    dataset = _fine_dataset().isel(latitude=slice(None, None, -1), longitude=slice(None, None, -1))
    target = _target_grid().isel(latitude=slice(None, None, -1)).rename(latitude="lat", longitude="lon")
    target.lat.attrs["standard_name"] = "latitude"
    target.lon.attrs["standard_name"] = "longitude"

    interpolated = interpolate_to_grid(dataset, target)

    numpy.testing.assert_array_equal(interpolated.latitude, target.lat)
    numpy.testing.assert_array_equal(interpolated.longitude, target.lon)
    numpy.testing.assert_allclose(interpolated.temperature, 3 * target.lat.values[:, None] + target.lon.values[None, :])


@pytest.mark.parametrize("latitude", [[45.0], [44.9, 45.0, 45.1], [44.9, 44.9 + 1 / 12, 45.1]])
def test_interpolate_to_grid_clears_source_resolution_when_the_target_is_unrecognised(latitude: list) -> None:
    target = _target_grid().drop_dims("latitude").assign_coords(latitude=latitude)

    interpolated = interpolate_to_grid(_fine_dataset(), target)

    source = get_dataset_source(interpolated)
    assert source == DatasetSource(kind="reference", name="ibi36", variant="copernicus")


def test_interpolate_to_grid_preserves_missing_provenance() -> None:
    dataset = xarray.Dataset({"temperature": _fine_dataset().temperature})
    assert get_dataset_source(interpolate_to_grid(dataset, _target_grid())) is None


def test_interpolate_to_grid_keeps_missing_stencil_values_missing() -> None:
    values = numpy.ones((4, 4))
    values[1, 1] = numpy.nan
    dataset = xarray.Dataset(
        {"temperature": (("latitude", "longitude"), values)},
        coords={"latitude": numpy.arange(4.0), "longitude": numpy.arange(4.0)},
    )
    target = xarray.Dataset(coords={"latitude": [0.5, 2.5], "longitude": [0.5, 2.5]})

    interpolated = interpolate_to_grid(dataset, target)

    numpy.testing.assert_allclose(interpolated.temperature, [[numpy.nan, 1.0], [1.0, 1.0]], equal_nan=True)
