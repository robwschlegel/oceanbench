# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import math
import re

import numpy
import pandas
import xarray

from oceanbench.core.classIV_support import OBSERVATION_COUNT_COLUMN
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension

_MESSAGE_COLUMN = "Message"
_LABEL_PATTERN = re.compile(r"^(.*?) \(([^)]*)\) \[([^\]]*)\]\{([^}]+)\} neighbourhood (\d+)x\d+(?: vs .+)?$")
_LEAD_DAY_PATTERN = re.compile(r"^Lead day (\d+)$")
_BOX_WIDTH_DECIMALS = 4


def _latitude_spacing_in_degrees(challenger_dataset: xarray.Dataset) -> float:
    latitudes = rename_dataset_with_standard_names(challenger_dataset)[Dimension.LATITUDE.key()].values
    return float(numpy.diff(latitudes.astype(numpy.float64)).mean())


def _json_number(value, number_type: type) -> float | int | None:
    return None if math.isnan(value) else number_type(value)


def neighbourhood_crps_score_rows(
    metric_function_name: str,
    table: pandas.DataFrame,
    challenger_dataset: xarray.Dataset,
) -> list[dict]:
    if _MESSAGE_COLUMN in table.columns:
        return []
    metric, reference = metric_function_name.split("_compared_to_")
    latitude_spacing = _latitude_spacing_in_degrees(challenger_dataset)
    lead_day_columns = {
        column: int(match.group(1)) for column in table.columns if (match := _LEAD_DAY_PATTERN.match(column))
    }
    rows = []
    for label in table.index:
        display_name, unit, standard_name, depth, box_size = _LABEL_PATTERN.match(label).groups()
        observation_fields = (
            {"observations": _json_number(table.at[label, OBSERVATION_COUNT_COLUMN], int)}
            if OBSERVATION_COUNT_COLUMN in table.columns
            else {}
        )
        rows += [
            {
                "metric": metric,
                "reference": reference,
                "variable": display_name.lower(),
                "standard_name": standard_name or None,
                "depth": depth,
                "unit": unit,
                "lead_day": lead_day,
                "value": _json_number(table.at[label, column], float),
                "box_size": int(box_size),
                "box_width_degrees": round(int(box_size) * latitude_spacing, _BOX_WIDTH_DECIMALS),
            }
            | observation_fields
            for column, lead_day in lead_day_columns.items()
        ]
    return rows
