# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray

from oceanbench.core.dataset_utils import Dimension, is_global_longitude_grid

OBSERVATION_DIMENSION = "observation"
NEIGHBOURHOOD_MEMBER_DIMENSION = "neighbourhood_member"
BOX_SIZE_DIMENSION = "box_size"
_BOX_ROW_DIMENSION = "box_row"
_BOX_COLUMN_DIMENSION = "box_column"


def _regular_grid_spacing(coordinate_values: numpy.ndarray) -> float:
    spacings = numpy.diff(coordinate_values)
    if not numpy.allclose(spacings, spacings[0]):
        raise ValueError("Neighbourhood CRPS needs a regular latitude and longitude grid.")
    return float(spacings[0])


def _nearest_grid_indices(
    coordinate_values: numpy.ndarray,
    target_values: xarray.DataArray,
) -> xarray.DataArray:
    spacing = _regular_grid_spacing(coordinate_values)
    return numpy.rint((target_values - coordinate_values[0]) / spacing).astype(numpy.int64)


def _box_offsets(box_size: int, dimension: str) -> xarray.DataArray:
    if box_size < 1 or box_size % 2 == 0:
        raise ValueError(f"Box size must be a positive odd number of grid cells, got {box_size}.")
    half_width = box_size // 2
    return xarray.DataArray(numpy.arange(-half_width, half_width + 1), dims=dimension)


def neighbourhood_values(
    field: xarray.DataArray,
    latitudes: xarray.DataArray,
    longitudes: xarray.DataArray,
    box_size: int,
) -> xarray.DataArray:
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    grid_latitudes = field[latitude_key].values
    grid_longitudes = field[longitude_key].values
    row_count = grid_latitudes.size
    column_count = grid_longitudes.size

    rows = _nearest_grid_indices(grid_latitudes, latitudes) + _box_offsets(box_size, _BOX_ROW_DIMENSION)
    columns = _nearest_grid_indices(grid_longitudes, longitudes) + _box_offsets(box_size, _BOX_COLUMN_DIMENSION)
    is_global = is_global_longitude_grid(grid_longitudes)
    wrapped_columns = columns % column_count if is_global else columns
    is_inside_grid = (rows >= 0) & (rows < row_count) & (wrapped_columns >= 0) & (wrapped_columns < column_count)

    box_values = field.isel(
        {
            latitude_key: rows.clip(0, row_count - 1),
            longitude_key: wrapped_columns.clip(0, column_count - 1),
        }
    ).where(is_inside_grid)
    return box_values.drop_vars([latitude_key, longitude_key], errors="ignore").stack(
        {NEIGHBOURHOOD_MEMBER_DIMENSION: [_BOX_ROW_DIMENSION, _BOX_COLUMN_DIMENSION]},
        create_index=False,
    )


def ensemble_crps(
    members: xarray.DataArray,
    observations: xarray.DataArray,
) -> xarray.DataArray:
    member_count = members.sizes[NEIGHBOURHOOD_MEMBER_DIMENSION]
    sorted_members = xarray.apply_ufunc(
        numpy.sort,
        members,
        input_core_dims=[[NEIGHBOURHOOD_MEMBER_DIMENSION]],
        output_core_dims=[[NEIGHBOURHOOD_MEMBER_DIMENSION]],
        kwargs={"axis": -1},
        dask="parallelized",
        output_dtypes=[members.dtype],
    )
    rank_weights = xarray.DataArray(
        2 * numpy.arange(1, member_count + 1) - member_count - 1,
        dims=NEIGHBOURHOOD_MEMBER_DIMENSION,
    )
    mean_absolute_error = abs(members - observations).mean(NEIGHBOURHOOD_MEMBER_DIMENSION, skipna=False)
    half_mean_member_difference = (sorted_members * rank_weights).sum(NEIGHBOURHOOD_MEMBER_DIMENSION, skipna=False) / (
        member_count**2
    )
    return mean_absolute_error - half_mean_member_difference


def neighbourhood_crps(
    field: xarray.DataArray,
    latitudes: xarray.DataArray,
    longitudes: xarray.DataArray,
    observations: xarray.DataArray,
    box_sizes: list[int],
) -> xarray.DataArray:
    largest_box_is_ocean = (
        neighbourhood_values(field, latitudes, longitudes, max(box_sizes)).notnull().all(NEIGHBOURHOOD_MEMBER_DIMENSION)
    )
    scores = [
        ensemble_crps(neighbourhood_values(field, latitudes, longitudes, box_size), observations)
        for box_size in box_sizes
    ]
    return xarray.concat(scores, dim=pandas.Index(box_sizes, name=BOX_SIZE_DIMENSION)).where(largest_box_is_ocean)
