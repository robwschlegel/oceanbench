# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import pytest
import xarray

from oceanbench.core.class4_neighbourhood_crps import (
    _box_sizes_on_challenger_grid,
    _crps_by_observation,
    _crps_table,
    _formatted_results,
)
from oceanbench.core.classIV_support import OBSERVATION_COUNT_COLUMN
from oceanbench.core.dataset_utils import Dimension, Variable

FIRST_DAYS = numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]")
DEPTHS = numpy.array([0.494, 1.541, 2.646])


def _challenger_grid(spacing: float) -> xarray.Dataset:
    return xarray.Dataset(
        coords={
            Dimension.LATITUDE.key(): numpy.arange(30) * spacing,
            Dimension.LONGITUDE.key(): numpy.arange(30) * spacing,
        }
    )


def _encoded_temperature() -> xarray.DataArray:
    first_day_index = xarray.DataArray(numpy.arange(2), dims=Dimension.FIRST_DAY_DATETIME.key())
    lead_day_index = xarray.DataArray(numpy.arange(3), dims=Dimension.LEAD_DAY_INDEX.key())
    depth_index = xarray.DataArray(numpy.arange(3), dims=Dimension.DEPTH.key())
    spatial_ones = xarray.DataArray(numpy.ones((12, 12)), dims=[Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()])
    encoded = 100.0 * first_day_index + 10.0 * lead_day_index + depth_index + 0 * spatial_ones
    return encoded.assign_coords(
        {
            Dimension.FIRST_DAY_DATETIME.key(): FIRST_DAYS,
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(3),
            Dimension.DEPTH.key(): DEPTHS,
            Dimension.LATITUDE.key(): numpy.arange(12) / 12,
            Dimension.LONGITUDE.key(): numpy.arange(12) / 12,
        }
    )


@pytest.mark.parametrize(
    ("spacing", "expected_box_sizes"),
    [(1 / 12, [1, 3, 5, 7]), (1 / 36, [3, 9, 15, 21])],
)
def test_box_sizes_cover_the_same_area_on_every_supported_grid(spacing: float, expected_box_sizes: list[int]):
    assert _box_sizes_on_challenger_grid(_challenger_grid(spacing)) == expected_box_sizes


@pytest.mark.parametrize("spacing", [0.25, 1 / 24])
def test_box_sizes_reject_grids_that_are_not_an_odd_fraction_of_twelfth_degree(spacing: float):
    with pytest.raises(ValueError, match="odd fraction"):
        _box_sizes_on_challenger_grid(_challenger_grid(spacing))


def test_each_observation_is_scored_against_its_own_forecast_lead_day_and_nearest_depth():
    observations_dataframe = pandas.DataFrame(
        {
            "observation_value": [0.0, 0.0, 0.0],
            Dimension.LATITUDE.key(): [0.5, 0.4, 0.3],
            Dimension.LONGITUDE.key(): [0.5, 0.4, 0.3],
            "first_day": pandas.to_datetime(["2024-01-03", "2024-01-10", "2024-01-10"]),
            "lead_day": [0, 2, 1],
            Dimension.DEPTH.key(): [0.4, 1.4, 2.9],
            "depth_bin": ["surface", "surface", "0-5m"],
        }
    )

    crps_by_observation = _crps_by_observation(
        _encoded_temperature(), observations_dataframe, box_sizes=[1, 3], minimum_ocean_fraction=0.5
    )

    single_cell = crps_by_observation[crps_by_observation["box_size"] == 1].sort_values("observation")
    assert single_cell["crps"].tolist() == [0.0, 121.0, 112.0]
    assert single_cell["depth_bin"].tolist() == ["surface", "surface", "0-5m"]


def test_formatted_results_have_one_row_per_depth_bin_and_box_size():
    crps_by_observation = pandas.DataFrame(
        {
            "box_size": [1, 3, 1, 3],
            "observation": [0, 0, 1, 1],
            "crps": [0.4, 0.2, 0.6, 0.4],
            "depth_bin": ["surface"] * 4,
            "lead_day": [0, 0, 0, 0],
        }
    )
    temperature_key = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()

    formatted = _formatted_results(_crps_table(crps_by_observation, temperature_key), lead_days_count=2)

    assert len(formatted) == 2
    assert formatted.index[1].endswith("{surface} neighbourhood 3x3")
    numpy.testing.assert_allclose(formatted.iloc[:, 0].values, [0.5, 0.3])
    assert formatted[OBSERVATION_COUNT_COLUMN].tolist() == [2, 2]
    assert formatted.iloc[:, 1].isnull().all()
