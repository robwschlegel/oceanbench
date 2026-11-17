# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import copernicusmarine
import numpy
import pandas
import xarray

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable

OSTIA_REPROCESSED_DATASET_IDENTIFIER = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
_KELVIN_TO_CELSIUS_OFFSET = 273.15
_SUBSET_MARGIN_DEGREES = 0.1


def _ostia_window(
    first_day_datetime: numpy.datetime64,
    lead_days_count: int,
    latitude_bounds: tuple[float, float],
    longitude_bounds: tuple[float, float],
) -> xarray.DataArray:
    first_day = pandas.Timestamp(first_day_datetime)
    last_day = first_day + pandas.Timedelta(days=lead_days_count - 1)
    dataset = copernicusmarine.open_dataset(
        dataset_id=OSTIA_REPROCESSED_DATASET_IDENTIFIER,
        variables=["analysed_sst", "sea_ice_fraction"],
        minimum_latitude=latitude_bounds[0],
        maximum_latitude=latitude_bounds[1],
        minimum_longitude=longitude_bounds[0],
        maximum_longitude=longitude_bounds[1],
        start_datetime=first_day.strftime("%Y-%m-%dT00:00:00"),
        end_datetime=last_day.strftime("%Y-%m-%dT00:00:00"),
    )
    ice_free_temperature = (dataset["analysed_sst"] - _KELVIN_TO_CELSIUS_OFFSET).where(
        dataset["sea_ice_fraction"].fillna(0) == 0
    )
    return ice_free_temperature.rename({"time": Dimension.LEAD_DAY_INDEX.key()}).assign_coords(
        {Dimension.LEAD_DAY_INDEX.key(): numpy.arange(lead_days_count)}
    )


def _bounds_with_margin(coordinate: xarray.DataArray) -> tuple[float, float]:
    return (
        float(coordinate.min()) - _SUBSET_MARGIN_DEGREES,
        float(coordinate.max()) + _SUBSET_MARGIN_DEGREES,
    )


def ostia_sea_surface_temperature_dataset(challenger_dataset: xarray.Dataset) -> xarray.Dataset:
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    first_day_datetimes = challenger[first_day_key].values
    windows = [
        _ostia_window(
            first_day_datetime,
            challenger.sizes[Dimension.LEAD_DAY_INDEX.key()],
            _bounds_with_margin(challenger[latitude_key]),
            _bounds_with_margin(challenger[longitude_key]),
        )
        for first_day_datetime in first_day_datetimes
    ]
    temperature = (
        xarray.concat(windows, dim=pandas.Index(first_day_datetimes, name=first_day_key))
        .interp(
            {
                latitude_key: challenger[latitude_key],
                longitude_key: challenger[longitude_key],
            },
            method="linear",
        )
        .astype(numpy.float32)
    )
    return xarray.Dataset({Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): temperature})
