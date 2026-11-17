# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray

from oceanbench.core.class1_neighbourhood_crps import crps_class1_validation
from oceanbench.core.classIV_support import OBSERVATION_COUNT_COLUMN
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.neighbourhood_crps_scores import neighbourhood_crps_score_rows

SCORES_FILE_ROW_FIELDS = {"metric", "reference", "variable", "standard_name", "depth", "unit", "lead_day", "value"}
OSTIA_METRIC_FUNCTION_NAME = "neighbourhood_crps_of_sea_surface_temperature_compared_to_ostia"


def _challenger(cells_per_degree: int, cells: int) -> xarray.Dataset:
    first_days = numpy.array(["2024-01-03"], dtype="datetime64[ns]")
    latitudes = 40 + numpy.arange(cells) / cells_per_degree
    longitudes = -10 + numpy.arange(cells) / cells_per_degree
    return xarray.Dataset(
        {
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.DEPTH.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                numpy.full((1, 2, 1, cells, cells), 15.0),
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): first_days,
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(2),
            Dimension.DEPTH.key(): numpy.array([0.494]),
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def _ostia_rows(challenger: xarray.Dataset) -> list[dict]:
    reference = challenger.isel({Dimension.DEPTH.key(): 0}, drop=True) + 0.5
    table = crps_class1_validation(challenger, reference, reference_name="OSTIA")
    return neighbourhood_crps_score_rows(OSTIA_METRIC_FUNCTION_NAME, table, challenger)


def test_class1_rows_carry_the_scores_file_fields_and_the_box_size():
    rows = _ostia_rows(_challenger(cells_per_degree=12, cells=12))

    assert len(rows) == 4 * 2
    assert all(SCORES_FILE_ROW_FIELDS | {"box_size", "box_width_degrees"} == set(row) for row in rows)
    numpy.testing.assert_allclose([row["value"] for row in rows], 0.5)
    assert {field: value for field, value in rows[0].items() if field != "value"} == {
        "metric": "neighbourhood_crps_of_sea_surface_temperature",
        "reference": "ostia",
        "variable": "surface temperature",
        "standard_name": "sea_water_potential_temperature",
        "depth": "surface",
        "unit": "°C",
        "lead_day": 1,
        "box_size": 1,
        "box_width_degrees": 0.0833,
    }
    assert [row["box_size"] for row in rows if row["lead_day"] == 2] == [1, 3, 5, 7]


def test_matched_boxes_of_a_one_thirty_sixth_degree_grid_have_the_same_widths_as_on_the_twelfth_degree_grid():
    twelfth_degree_widths = [row["box_width_degrees"] for row in _ostia_rows(_challenger(12, 12))]
    thirty_sixth_degree_rows = _ostia_rows(_challenger(36, 30))

    assert [row["box_size"] for row in thirty_sixth_degree_rows if row["lead_day"] == 1] == [3, 9, 15, 21]
    assert [row["box_width_degrees"] for row in thirty_sixth_degree_rows] == twelfth_degree_widths


def test_class4_rows_repeat_the_observation_count_and_write_missing_scores_as_null():
    label = "Temperature (°C) [sea_water_potential_temperature]{0-5m} neighbourhood 3x3"
    table = pandas.DataFrame(
        {"Lead day 1": [0.25], "Lead day 2": [numpy.nan], OBSERVATION_COUNT_COLUMN: [90]}, index=[label]
    )

    rows = neighbourhood_crps_score_rows(
        "neighbourhood_crps_of_temperature_compared_to_observations", table, _challenger(12, 12)
    )

    assert [(row["lead_day"], row["value"], row["observations"]) for row in rows] == [(1, 0.25, 90), (2, None, 90)]
    assert {row["reference"] for row in rows} == {"observations"}
    assert {row["depth"] for row in rows} == {"0-5m"}


def test_a_metric_that_could_not_be_computed_has_no_rows():
    table = pandas.DataFrame({"Message": ["No observations for these dates."]})

    assert neighbourhood_crps_score_rows(OSTIA_METRIC_FUNCTION_NAME, table, _challenger(12, 12)) == []
