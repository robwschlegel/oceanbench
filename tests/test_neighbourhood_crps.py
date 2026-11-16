# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
import xarray

from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.neighbourhood_crps import (
    BOX_SIZE_DIMENSION,
    NEIGHBOURHOOD_MEMBER_DIMENSION,
    OBSERVATION_DIMENSION,
    ensemble_crps,
    gridded_neighbourhood_crps,
    neighbourhood_crps,
    neighbourhood_values,
)


def _field(latitudes: numpy.ndarray, longitudes: numpy.ndarray, values: numpy.ndarray) -> xarray.DataArray:
    return xarray.DataArray(
        values,
        dims=[Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={Dimension.LATITUDE.key(): latitudes, Dimension.LONGITUDE.key(): longitudes},
    )


def _along_observations(values: list[float]) -> xarray.DataArray:
    return xarray.DataArray(numpy.array(values, dtype=float), dims=OBSERVATION_DIMENSION)


def _brute_force_crps(members: numpy.ndarray, observation: float) -> float:
    member_count = members.size
    pairwise_differences = numpy.abs(members[:, None] - members[None, :]).sum()
    return numpy.abs(members - observation).mean() - pairwise_differences / (2 * member_count**2)


def test_ensemble_crps_matches_hand_worked_example():
    members = xarray.DataArray([[1.0, 2.0, 4.0]], dims=[OBSERVATION_DIMENSION, NEIGHBOURHOOD_MEMBER_DIMENSION])

    crps = ensemble_crps(members, _along_observations([3.0]))

    numpy.testing.assert_allclose(crps.values, [2.0 / 3.0])


def test_ensemble_crps_matches_pairwise_definition():
    random_generator = numpy.random.default_rng(seed=0)
    member_values = random_generator.normal(size=(20, 49))
    observation_values = random_generator.normal(size=20)
    members = xarray.DataArray(member_values, dims=[OBSERVATION_DIMENSION, NEIGHBOURHOOD_MEMBER_DIMENSION])

    crps = ensemble_crps(members, _along_observations(list(observation_values)))

    expected = [_brute_force_crps(member_values[index], observation_values[index]) for index in range(20)]
    numpy.testing.assert_allclose(crps.values, expected)


def test_single_cell_box_gives_absolute_error_of_nearest_cell():
    random_generator = numpy.random.default_rng(seed=1)
    latitudes = numpy.arange(-5.0, 5.0, 0.5)
    longitudes = numpy.arange(10.0, 20.0, 0.5)
    field = _field(latitudes, longitudes, random_generator.normal(size=(latitudes.size, longitudes.size)))
    observation_latitudes = _along_observations([-2.1, 0.2, 3.4])
    observation_longitudes = _along_observations([12.1, 15.3, 17.8])
    observations = _along_observations([0.5, -0.3, 1.2])

    crps = neighbourhood_crps(field, observation_latitudes, observation_longitudes, observations, box_sizes=[1])

    nearest_values = field.sel(
        {
            Dimension.LATITUDE.key(): observation_latitudes,
            Dimension.LONGITUDE.key(): observation_longitudes,
        },
        method="nearest",
    )
    numpy.testing.assert_allclose(crps.sel({BOX_SIZE_DIMENSION: 1}).values, abs(nearest_values - observations).values)


def test_displaced_sharp_front_scores_better_with_a_neighbourhood():
    latitudes = numpy.arange(0.0, 20.0)
    longitudes = numpy.arange(0.0, 20.0)
    displaced_front = (longitudes >= 11).astype(float)
    field = _field(latitudes, longitudes, numpy.tile(displaced_front, (latitudes.size, 1)))
    observation_longitudes = _along_observations(list(numpy.arange(6.0, 15.0)))
    observation_latitudes = xarray.full_like(observation_longitudes, 10.0)
    observations = (observation_longitudes >= 10).astype(float)

    mean_crps = neighbourhood_crps(
        field, observation_latitudes, observation_longitudes, observations, box_sizes=[1, 3]
    ).mean(OBSERVATION_DIMENSION)

    assert mean_crps.sel({BOX_SIZE_DIMENSION: 3}) < mean_crps.sel({BOX_SIZE_DIMENSION: 1})


def _field_with_land_cells(land_cells: list[tuple[int, int]]) -> xarray.DataArray:
    latitudes = numpy.arange(0.0, 10.0)
    longitudes = numpy.arange(0.0, 10.0)
    values = numpy.arange(100.0).reshape(10, 10)
    for row, column in land_cells:
        values[row, column] = numpy.nan
    return _field(latitudes, longitudes, values)


def test_ensemble_crps_leaves_out_land_members():
    members = xarray.DataArray(
        [[1.0, numpy.nan, 2.0, 4.0, numpy.nan]], dims=[OBSERVATION_DIMENSION, NEIGHBOURHOOD_MEMBER_DIMENSION]
    )

    crps = ensemble_crps(members, _along_observations([3.0]))

    numpy.testing.assert_allclose(crps.values, [2.0 / 3.0])


def test_partly_land_box_is_scored_on_its_ocean_cells_by_default():
    field = _field_with_land_cells([(5, 6)])

    crps = neighbourhood_crps(
        field, _along_observations([5.0]), _along_observations([5.0]), _along_observations([0.0]), box_sizes=[1, 3]
    )

    ocean_cells = numpy.delete(field.values[4:7, 4:7].ravel(), 5)
    numpy.testing.assert_allclose(crps.sel({BOX_SIZE_DIMENSION: 3}).values, [_brute_force_crps(ocean_cells, 0.0)])


def test_observation_with_a_land_centre_cell_is_not_scored():
    crps = neighbourhood_crps(
        _field_with_land_cells([(5, 5)]),
        _along_observations([5.0]),
        _along_observations([5.0]),
        _along_observations([0.0]),
        box_sizes=[1, 3],
    )

    assert crps.isnull().all()


def test_observation_is_dropped_at_every_box_size_when_one_box_has_too_little_ocean():
    mostly_land_around_centre = [(row, column) for row in (4, 5, 6) for column in (4, 5, 6) if (row, column) != (5, 5)]
    observations = _along_observations([0.0, 0.0])

    crps = neighbourhood_crps(
        _field_with_land_cells(mostly_land_around_centre),
        _along_observations([5.0, 2.0]),
        _along_observations([5.0, 8.0]),
        observations,
        box_sizes=[1, 3],
    )

    assert crps.isel({OBSERVATION_DIMENSION: 0}).isnull().all()
    assert crps.isel({OBSERVATION_DIMENSION: 1}).notnull().all()


def test_full_ocean_fraction_keeps_only_boxes_that_are_entirely_ocean():
    crps = neighbourhood_crps(
        _field_with_land_cells([(5, 7)]),
        _along_observations([5.0, 5.0]),
        _along_observations([5.0, 2.0]),
        _along_observations([1.0, 1.0]),
        box_sizes=[1, 3, 5],
        minimum_ocean_fraction=1.0,
    )

    assert crps.isel({OBSERVATION_DIMENSION: 0}).isnull().all()
    assert crps.isel({OBSERVATION_DIMENSION: 1}).notnull().all()


@pytest.mark.parametrize("minimum_ocean_fraction", [0.0, 1.5])
def test_minimum_ocean_fraction_must_be_a_proportion(minimum_ocean_fraction: float):
    with pytest.raises(ValueError, match="Minimum ocean fraction"):
        neighbourhood_crps(
            _field_with_land_cells([]),
            _along_observations([5.0]),
            _along_observations([5.0]),
            _along_observations([0.0]),
            box_sizes=[1],
            minimum_ocean_fraction=minimum_ocean_fraction,
        )


def test_box_wraps_across_the_dateline_on_a_global_grid():
    latitudes = numpy.arange(-10.0, 10.0, 1.0)
    longitudes = numpy.arange(-180.0, 180.0, 1.0)
    longitude_values = numpy.tile(longitudes, (latitudes.size, 1))
    field = _field(latitudes, longitudes, longitude_values)

    box = neighbourhood_values(field, _along_observations([0.0]), _along_observations([179.2]), box_size=3)

    assert sorted(numpy.unique(box.values)) == [-180.0, 178.0, 179.0]


def test_box_beyond_a_regional_grid_edge_is_missing():
    latitudes = numpy.arange(0.0, 10.0)
    longitudes = numpy.arange(0.0, 10.0)
    field = _field(latitudes, longitudes, numpy.ones((latitudes.size, longitudes.size)))

    box = neighbourhood_values(field, _along_observations([5.0]), _along_observations([9.0]), box_size=3)

    assert box.isnull().sum() == 3


@pytest.mark.parametrize("box_size", [0, 2, -3])
def test_box_size_must_be_positive_and_odd(box_size: int):
    field = _field(numpy.arange(3.0), numpy.arange(3.0), numpy.ones((3, 3)))

    with pytest.raises(ValueError, match="positive odd"):
        neighbourhood_values(field, _along_observations([1.0]), _along_observations([1.0]), box_size=box_size)


def _grid_cell_centres(field: xarray.DataArray) -> tuple[xarray.DataArray, xarray.DataArray]:
    stacked = field.stack({OBSERVATION_DIMENSION: [Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()]})
    return (
        xarray.DataArray(stacked[Dimension.LATITUDE.key()].values, dims=OBSERVATION_DIMENSION),
        xarray.DataArray(stacked[Dimension.LONGITUDE.key()].values, dims=OBSERVATION_DIMENSION),
    )


@pytest.mark.parametrize(
    "longitudes",
    [numpy.arange(10.0, 22.0), numpy.arange(-180.0, 180.0, 30.0)],
    ids=["regional", "global"],
)
def test_gridded_crps_equals_the_point_crps_at_every_grid_cell(longitudes: numpy.ndarray):
    random_generator = numpy.random.default_rng(seed=3)
    latitudes = numpy.arange(-5.0, 5.0)
    values = random_generator.normal(size=(latitudes.size, longitudes.size))
    values[2, 3] = numpy.nan
    values[6, 0] = numpy.nan
    field = _field(latitudes, longitudes, values)
    reference = _field(latitudes, longitudes, random_generator.normal(size=values.shape))

    gridded = gridded_neighbourhood_crps(field, reference, box_sizes=[1, 3, 5])

    observation_latitudes, observation_longitudes = _grid_cell_centres(field)
    reference_values = xarray.DataArray(
        reference.stack({OBSERVATION_DIMENSION: [Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()]}).values,
        dims=OBSERVATION_DIMENSION,
    )
    point = neighbourhood_crps(field, observation_latitudes, observation_longitudes, reference_values, [1, 3, 5])
    numpy.testing.assert_allclose(
        gridded.transpose(BOX_SIZE_DIMENSION, Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()).values.reshape(
            3, -1
        ),
        point.transpose(BOX_SIZE_DIMENSION, OBSERVATION_DIMENSION).values,
    )


def test_gridded_crps_skips_cells_without_a_reference_value():
    field = _field_with_land_cells([])
    reference = _field_with_land_cells([(4, 4)])

    gridded = gridded_neighbourhood_crps(field, reference, box_sizes=[1, 3])

    assert gridded.isel({Dimension.LATITUDE.key(): 4, Dimension.LONGITUDE.key(): 4}).isnull().all()
    assert gridded.isel({Dimension.LATITUDE.key(): 5, Dimension.LONGITUDE.key(): 5}).notnull().all()
