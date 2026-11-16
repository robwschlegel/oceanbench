# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray

from oceanbench.core.class4_neighbourhood_crps import box_sizes_on_challenger_grid
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import VARIABLE_METADATA, Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.neighbourhood_crps import BOX_SIZE_DIMENSION, gridded_neighbourhood_crps


def _surface_temperature(dataset: xarray.Dataset) -> xarray.DataArray:
    temperature = dataset[Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()]
    if Dimension.DEPTH.key() not in temperature.dims:
        return temperature
    return temperature.isel({Dimension.DEPTH.key(): 0}, drop=True)


def _area_weighted_mean_by_lead_day(crps: xarray.DataArray) -> xarray.DataArray:
    area_weights = numpy.cos(numpy.deg2rad(crps[Dimension.LATITUDE.key()]))
    return crps.weighted(area_weights).mean(
        [Dimension.FIRST_DAY_DATETIME.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()]
    )


def _label(reference_name: str, box_size: int) -> str:
    variable_key = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
    display_name, unit = VARIABLE_METADATA[variable_key]
    return (
        f"Surface {display_name.lower()} ({unit}) [{variable_key}]{{surface}} "
        f"neighbourhood {box_size}x{box_size} vs {reference_name}"
    )


def _formatted_results(
    mean_crps: xarray.DataArray,
    reference_name: str,
    lead_days_count: int,
) -> pandas.DataFrame:
    table = mean_crps.transpose(BOX_SIZE_DIMENSION, Dimension.LEAD_DAY_INDEX.key()).to_pandas()
    return table.set_axis([_label(reference_name, box_size) for box_size in table.index], axis="index").set_axis(
        lead_day_labels(1, lead_days_count), axis="columns"
    )


def crps_class1_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    reference_name: str,
    minimum_ocean_fraction: float = 0.5,
) -> pandas.DataFrame:
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    crps = gridded_neighbourhood_crps(
        _surface_temperature(challenger),
        _surface_temperature(reference_dataset),
        box_sizes_on_challenger_grid(challenger),
        minimum_ocean_fraction,
    )
    return _formatted_results(
        _area_weighted_mean_by_lead_day(crps).compute(),
        reference_name,
        challenger.sizes[Dimension.LEAD_DAY_INDEX.key()],
    ).rename_axis(index=None, columns=None)
