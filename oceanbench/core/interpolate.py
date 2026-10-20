# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import xarray
import numpy
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names, StandardDimension
from oceanbench.core.dataset_source import (
    DATASET_SOURCE_RESOLUTION_ATTRIBUTE,
    get_dataset_source,
    with_dataset_source,
)
from oceanbench.core.resolution import get_dataset_resolution


def interpolate_1_degree(data: xarray.Dataset) -> xarray.Dataset:
    data = rename_dataset_with_standard_names(data)

    latitude_dimension = StandardDimension.LATITUDE.value
    longitude_dimension = StandardDimension.LONGITUDE.value
    time_dimension = StandardDimension.TIME.value
    depth_dimension = StandardDimension.DEPTH.value

    latitude_minimum = data[latitude_dimension].min().values
    latitude_maximum = data[latitude_dimension].max().values
    longitude_minimum = data[longitude_dimension].min().values
    longitude_maximum = data[longitude_dimension].max().values

    latitude_start = numpy.ceil(latitude_minimum - 0.5) + 0.5
    latitude_end = numpy.floor(latitude_maximum + 0.5) - 0.5
    longitude_start = numpy.ceil(longitude_minimum - 0.5) + 0.5
    longitude_end = numpy.floor(longitude_maximum + 0.5) - 0.5

    new_latitude = numpy.arange(latitude_start, latitude_end + 1, 1.0)
    new_longitude = numpy.arange(longitude_start, longitude_end + 1, 1.0)

    chunk_dimensions = {latitude_dimension: -1, longitude_dimension: -1}
    if time_dimension in data.dims:
        chunk_dimensions[time_dimension] = 1
    if depth_dimension in data.dims:
        chunk_dimensions[depth_dimension] = 1

    data = data.chunk(chunk_dimensions)
    interpolated = data.interp(**{latitude_dimension: new_latitude, longitude_dimension: new_longitude})

    dataset_source = get_dataset_source(data)
    if dataset_source is None:
        return interpolated

    return with_dataset_source(
        interpolated,
        kind=dataset_source.kind,
        name=dataset_source.name,
        resolution="one_degree",
    )


def _horizontal_coordinate(dataset: xarray.Dataset, name: str) -> numpy.ndarray:
    if name not in dataset.coords or dataset[name].dims != (name,):
        raise ValueError(f"Interpolation requires a 1D {name} coordinate.")
    values = dataset[name].values
    if not values.size or not numpy.isfinite(values).all():
        raise ValueError(f"Interpolation requires a finite, nonempty {name} coordinate.")
    differences = numpy.diff(values.astype(float))
    if not ((differences > 0).all() or (differences < 0).all()):
        raise ValueError(f"Interpolation requires strictly monotonic {name} coordinates.")
    return values


def _target_resolution(target_grid: xarray.Dataset) -> str | None:
    for name in (StandardDimension.LATITUDE.value, StandardDimension.LONGITUDE.value):
        differences = numpy.diff(target_grid[name].values)
        if not differences.size or not numpy.allclose(differences, differences[0], rtol=1e-3, atol=1e-6):
            return None
    try:
        return get_dataset_resolution(target_grid)
    except ValueError:
        return None


def interpolate_to_grid(data: xarray.Dataset, target_grid: xarray.Dataset) -> xarray.Dataset:
    """Linearly interpolate lazily to the target's exact 1D horizontal coordinates.

    Target coordinates outside the source domain raise ValueError.
    """
    data = rename_dataset_with_standard_names(data)
    target_grid = rename_dataset_with_standard_names(target_grid)
    horizontal_dimensions = (StandardDimension.LATITUDE.value, StandardDimension.LONGITUDE.value)
    source_windows = {}
    for name in horizontal_dimensions:
        source_values = _horizontal_coordinate(data, name)
        target_values = _horizontal_coordinate(target_grid, name)
        if source_values.size < 2:
            raise ValueError(f"Linear interpolation requires at least two source {name} points.")
        if source_values[0] > source_values[-1]:
            data = data.sortby(name)
            source_values = source_values[::-1]
        if target_values.min() < source_values[0] or target_values.max() > source_values[-1]:
            raise ValueError(f"Target {name} coordinates extend outside the source grid; extrapolation is disabled.")
        start = max(int(numpy.searchsorted(source_values, target_values.min(), side="left")) - 1, 0)
        stop = min(int(numpy.searchsorted(source_values, target_values.max(), side="right")) + 1, source_values.size)
        source_windows[name] = slice(start, stop)

    chunk_dimensions = {name: -1 for name in horizontal_dimensions}
    for name in (StandardDimension.TIME.value, StandardDimension.DEPTH.value):
        if name in data.dims:
            chunk_dimensions[name] = 1
    interpolated = (
        data.isel(source_windows)
        .chunk(chunk_dimensions)
        .interp({name: target_grid[name] for name in horizontal_dimensions}, method="linear", assume_sorted=True)
    )
    interpolated = interpolated.drop_attrs(deep=False).assign_attrs(
        {name: value for name, value in interpolated.attrs.items() if name != DATASET_SOURCE_RESOLUTION_ATTRIBUTE}
    )
    dataset_source = get_dataset_source(data)
    if dataset_source is None:
        return interpolated
    return with_dataset_source(
        interpolated,
        kind=dataset_source.kind,
        name=dataset_source.name,
        resolution=_target_resolution(target_grid),
        variant=dataset_source.variant,
    )
