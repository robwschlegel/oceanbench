# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from contextlib import ExitStack

import copernicusmarine
import numpy
import pandas
import xarray

from oceanbench.core.dataset_source import with_dataset_source
from oceanbench.core.dataset_utils import Dimension, SPATIAL_COORDINATE_ALIGNMENT_ATOL, Variable


_COORDINATE_UNITS = {
    Dimension.DEPTH.key(): "m",
    Dimension.LATITUDE.key(): "degrees_north",
    Dimension.LONGITUDE.key(): "degrees_east",
}
_VARIABLES = {
    "thetao": (Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(), "degrees_C"),
    "so": (Variable.SEA_WATER_SALINITY.key(), "1e-3"),
    "uo": (Variable.EASTWARD_SEA_WATER_VELOCITY.key(), "m/s"),
    "vo": (Variable.NORTHWARD_SEA_WATER_VELOCITY.key(), "m/s"),
    "ssh": (Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(), "m"),
}
_DATASET_SOURCES = {
    "thetao": ("cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m", ["thetao"]),
    "so": ("cmems_mod_ibi_phy-sal_my_0.027deg_P1D-m", ["so"]),
    "currents": ("cmems_mod_ibi_phy-cur_my_0.027deg_P1D-m", ["uo", "vo"]),
    "ssh": ("cmems_mod_ibi_phy-ssh_my_0.027deg_P1D-m", ["zos"]),
}
_DATASET_VERSION = "202511"


def _timestamp(value: object) -> pandas.Timestamp:
    timestamp = pandas.Timestamp(value)
    if pandas.isna(timestamp) or timestamp.tzinfo is not None:
        raise ValueError("IBI36 dates must be finite timestamps without a timezone.")
    return timestamp


def _axis_values(coordinate: xarray.DataArray, name: str) -> numpy.ndarray:
    values = numpy.asarray(coordinate.values)
    if values.ndim != 1 or not values.size or not numpy.isfinite(values).all():
        raise ValueError(f"IBI36 {name} must be a finite, nonempty 1D coordinate.")
    steps = numpy.diff(values.astype(float))
    if steps.size and not ((steps > 0).all() or (steps < 0).all()):
        raise ValueError(f"IBI36 {name} must be strictly monotonic without duplicates.")
    expected_units = _COORDINATE_UNITS[name]
    accepted_units = {expected_units, expected_units.replace("degrees", "degree")}
    if name == Dimension.DEPTH.key():
        accepted_units |= {"metres", "meters"}
        if (values < 0).any() or coordinate.attrs.get("positive", "down") != "down":
            raise ValueError("IBI36 depths must be nonnegative metres, positive down.")
    if coordinate.attrs.get("units", expected_units) not in accepted_units:
        raise ValueError(f"IBI36 {name} units must be {expected_units}.")
    return values


def _normalise_variable(variable: xarray.DataArray, name: str) -> xarray.DataArray:
    standard_name, units = _VARIABLES[name]
    accepted_names = {standard_name} | (
        {"sea_water_practical_salinity", "sea_water_salinity"} if name == "so" else set()
    )
    if variable.attrs.get("standard_name", standard_name) not in accepted_names:
        raise ValueError(f"IBI36 {name} has incompatible standard_name.")
    accepted_units = {
        "degrees_C": {"degrees_C", "degree_Celsius", "degrees_Celsius", "degC", "degree_C", "Celsius"},
        "1e-3": {"1e-3", "1e-03", "0.001", "PSU", "psu"},
        "m": {"m", "metres", "meters"},
        "m/s": {"m/s", "m s-1", "m s^-1", "m s**-1"},
    }
    if variable.attrs.get("units", units) not in accepted_units[units]:
        raise ValueError(f"IBI36 {name} has unsupported units.")
    return variable.where(numpy.isfinite(variable) & (abs(variable) < 1e30)).assign_attrs(
        standard_name=standard_name,
        units=units,
        ibi36_source_variable=name,
        ibi36_source_standard_name=variable.attrs.get(
            "standard_name", "sea_water_practical_salinity" if name == "so" else standard_name
        ),
        ibi36_units_basis="source variable" if "units" in variable.attrs else "native IBI36 dataset metadata",
    )


def _normalise_dataset(source: xarray.Dataset, days: pandas.DatetimeIndex) -> xarray.Dataset:
    for name in _COORDINATE_UNITS:
        if name not in source.coords or source[name].dims != (name,):
            raise ValueError(f"IBI36 requires {name}({name}) coordinates.")
    _validate_source_times(source, days)
    missing = set(_VARIABLES).difference(source.data_vars)
    if missing:
        raise ValueError(f"IBI36 datasets are missing variables: {', '.join(sorted(missing))}.")
    dataset = source[list(_VARIABLES)]
    for name in _COORDINATE_UNITS:
        values = _axis_values(dataset[name], name)
        if values.size > 1 and values[0] > values[-1]:
            dataset = dataset.sortby(name)
    horizontal_dimensions = (Dimension.LATITUDE.key(), Dimension.LONGITUDE.key())
    for name in _VARIABLES:
        expected = (Dimension.TIME.key(),) + (() if name == "ssh" else (Dimension.DEPTH.key(),)) + horizontal_dimensions
        if set(dataset[name].dims) != set(expected) or len(dataset[name].dims) != len(expected):
            raise ValueError(f"IBI36 {name} dimensions must be {expected}.")
        dataset[name] = dataset[name].transpose(*expected)
    return (
        dataset.assign({name: _normalise_variable(dataset[name], name) for name in _VARIABLES})
        .rename({name: metadata[0] for name, metadata in _VARIABLES.items()})
        .assign_coords(
            {
                name: dataset[name].assign_attrs(
                    standard_name=name, units=units, **({"positive": "down"} if name == Dimension.DEPTH.key() else {})
                )
                for name, units in _COORDINATE_UNITS.items()
            }
        )
    )


def _validate_source_times(source: xarray.Dataset, days: pandas.DatetimeIndex) -> None:
    if "time" not in source.coords or source["time"].dims != ("time",):
        raise ValueError("IBI36 requires a time coordinate.")
    times = source["time"].values
    if not numpy.issubdtype(times.dtype, numpy.datetime64) or not numpy.array_equal(
        times, days.values.astype("datetime64[ns]")
    ):
        raise ValueError("IBI36 daily dataset timestamps must match the requested dates at midnight.")


def _ocean_masks(source: xarray.Dataset, reference: xarray.Dataset) -> tuple[xarray.DataArray, xarray.DataArray]:
    if "mask" not in source or set(source["mask"].dims) != set(_COORDINATE_UNITS):
        raise ValueError("IBI36 bathymetry requires mask(depth, latitude, longitude).")
    mask = source["mask"].transpose(*_COORDINATE_UNITS).sortby(list(_COORDINATE_UNITS))
    for name in _COORDINATE_UNITS:
        values = _axis_values(mask[name], name)
        if name == Dimension.DEPTH.key():
            continue
        if values.size != reference.sizes[name] or not numpy.allclose(
            values, reference[name].values, rtol=0, atol=SPATIAL_COORDINATE_ALIGNMENT_ATOL
        ):
            raise ValueError(f"IBI36 bathymetry {name} does not match the data grid.")
    mask = mask.assign_coords({name: reference[name] for name in (Dimension.LATITUDE.key(), Dimension.LONGITUDE.key())})
    wet_cells = numpy.isfinite(mask) & (mask > 0) & (mask < 1e30)
    surface_mask = wet_cells.isel({Dimension.DEPTH.key(): 0}, drop=True)
    try:
        depth_mask = wet_cells.sel(
            {Dimension.DEPTH.key(): reference[Dimension.DEPTH.key()]}, method="nearest", tolerance=1e-3
        )
    except KeyError as error:
        raise ValueError("IBI36 bathymetry depth does not match the data grid.") from error
    return depth_mask.assign_coords({Dimension.DEPTH.key(): reference[Dimension.DEPTH.key()]}), surface_mask


def ibi36_reference(
    start_date: str,
    end_date: str,
    username: str | None = None,
    password: str | None = None,
    minimum_longitude: float | None = None,
    maximum_longitude: float | None = None,
    minimum_latitude: float | None = None,
    maximum_latitude: float | None = None,
    minimum_depth: float | None = None,
    maximum_depth: float | None = None,
) -> xarray.Dataset:
    first_day, last_day = _timestamp(start_date).normalize(), _timestamp(end_date).normalize()
    if first_day > last_day:
        raise ValueError("IBI36 end_date must be on or after start_date.")
    limits = (
        (minimum_longitude, maximum_longitude, "longitude"),
        (minimum_latitude, maximum_latitude, "latitude"),
        (minimum_depth, maximum_depth, "depth"),
    )
    for minimum, maximum, name in limits:
        if minimum is not None and not numpy.isfinite(minimum):
            raise ValueError(f"IBI36 minimum {name} must be finite.")
        if maximum is not None and not numpy.isfinite(maximum):
            raise ValueError(f"IBI36 maximum {name} must be finite.")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"IBI36 minimum {name} must not exceed maximum {name}.")
    if (minimum_depth is not None and minimum_depth < 0) or (maximum_depth is not None and maximum_depth < 0):
        raise ValueError("IBI36 depth limits must be nonnegative metres, positive down.")

    subset_parameters = {
        key: value
        for key, value in (
            ("minimum_longitude", minimum_longitude),
            ("maximum_longitude", maximum_longitude),
            ("minimum_latitude", minimum_latitude),
            ("maximum_latitude", maximum_latitude),
            ("minimum_depth", minimum_depth),
            ("maximum_depth", maximum_depth),
        )
        if value is not None
    }
    days = pandas.date_range(first_day, last_day, freq="D")
    with ExitStack() as sources:
        datasets = []
        for name, (dataset_id, source_names) in _DATASET_SOURCES.items():
            variable_subset = {
                key: value
                for key, value in subset_parameters.items()
                if name != "ssh" or key not in {"minimum_depth", "maximum_depth"}
            }
            parameters = {
                "dataset_id": dataset_id,
                "dataset_version": _DATASET_VERSION,
                "variables": source_names,
                "start_datetime": first_day.strftime("%Y-%m-%dT00:00:00"),
                "end_datetime": last_day.strftime("%Y-%m-%dT00:00:00"),
                "username": username,
                "password": password,
                "coordinates_selection_method": "inside",
                **variable_subset,
            }
            if name != "ssh":
                parameters["vertical_axis"] = "depth"
            source = copernicusmarine.open_dataset(
                **parameters,
            )
            sources.callback(source.close)
            _validate_source_times(source, days)
            datasets.append(source.rename({"zos": "ssh"} if name == "ssh" else {}).chunk())
        dataset = _normalise_dataset(xarray.merge(datasets, compat="equals", join="exact"), days)

        mask_source = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_ibi_phy_my_0.027deg-3D_static",
            dataset_part="bathy",
            dataset_version=_DATASET_VERSION,
            variables=["mask"],
            vertical_axis="depth",
            username=username,
            password=password,
            coordinates_selection_method="inside",
            **{key: value for key, value in subset_parameters.items() if key != "minimum_depth"},
        )
        sources.callback(mask_source.close)
        depth_mask, surface_mask = _ocean_masks(mask_source.chunk(), dataset)
        dataset = dataset.assign(
            {
                name: variable.where(depth_mask if Dimension.DEPTH.key() in variable.dims else surface_mask)
                for name, variable in dataset.data_vars.items()
            }
        ).assign_attrs(
            ibi36_source_datasets={name: dataset_id for name, (dataset_id, _) in _DATASET_SOURCES.items()},
            ibi36_mask_source="cmems_mod_ibi_phy_my_0.027deg-3D_static:bathy",
            ibi36_native_mask_applied=True,
            ibi36_source_product="IBI_MULTIYEAR_PHY_005_002",
            ibi36_dataset_version=_DATASET_VERSION,
        )
        dataset = with_dataset_source(dataset, kind="reference", name="ibi36", resolution="thirty_sixth_degree")
        dataset.set_close(sources.pop_all().close)
        return dataset
