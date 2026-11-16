# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import copernicusmarine
import numpy
import pandas
import xarray

from oceanbench.core.dataset_utils import Dimension

L3_SEA_SURFACE_TEMPERATURE_DATASET_IDENTIFIER = "cmems_obs-sst_glo_phy_my_l3s_P1D-m"
MINIMUM_QUALITY_LEVEL = 4
_KELVIN_TO_CELSIUS_OFFSET = 273.15
_SUBSET_MARGIN_DEGREES = 0.1


def _bounds_with_margin(coordinate: xarray.DataArray) -> tuple[float, float]:
    return (
        float(coordinate.min()) - _SUBSET_MARGIN_DEGREES,
        float(coordinate.max()) + _SUBSET_MARGIN_DEGREES,
    )


def _l3_window(
    first_day_datetime: numpy.datetime64,
    lead_days_count: int,
    latitude_bounds: tuple[float, float],
    longitude_bounds: tuple[float, float],
) -> xarray.DataArray:
    first_day = pandas.Timestamp(first_day_datetime)
    last_day = first_day + pandas.Timedelta(days=lead_days_count - 1)
    dataset = copernicusmarine.open_dataset(
        dataset_id=L3_SEA_SURFACE_TEMPERATURE_DATASET_IDENTIFIER,
        variables=["sea_surface_temperature", "quality_level"],
        minimum_latitude=latitude_bounds[0],
        maximum_latitude=latitude_bounds[1],
        minimum_longitude=longitude_bounds[0],
        maximum_longitude=longitude_bounds[1],
        start_datetime=first_day.strftime("%Y-%m-%dT00:00:00"),
        end_datetime=last_day.strftime("%Y-%m-%dT00:00:00"),
    )
    good_quality_temperature = (dataset["sea_surface_temperature"] - _KELVIN_TO_CELSIUS_OFFSET).where(
        dataset["quality_level"] >= MINIMUM_QUALITY_LEVEL
    )
    return good_quality_temperature.rename({"time": Dimension.LEAD_DAY_INDEX.key()}).assign_coords(
        {Dimension.LEAD_DAY_INDEX.key(): numpy.arange(lead_days_count)}
    )


def l3_sea_surface_temperature_windows(challenger_dataset: xarray.Dataset) -> list[xarray.DataArray]:
    return [
        _l3_window(
            first_day_datetime,
            challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()],
            _bounds_with_margin(challenger_dataset[Dimension.LATITUDE.key()]),
            _bounds_with_margin(challenger_dataset[Dimension.LONGITUDE.key()]),
        )
        for first_day_datetime in challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    ]


def l3_day_observations(
    window: xarray.DataArray,
    first_day_datetime: numpy.datetime64,
    lead_day: int,
) -> pandas.DataFrame:
    day = window.isel({Dimension.LEAD_DAY_INDEX.key(): lead_day}).compute()
    latitudes, longitudes = numpy.meshgrid(
        day[Dimension.LATITUDE.key()].values, day[Dimension.LONGITUDE.key()].values, indexing="ij"
    )
    values = day.transpose(Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()).values
    is_valid = numpy.isfinite(values)
    return pandas.DataFrame(
        {
            "observation_value": values[is_valid].astype(numpy.float64),
            Dimension.LATITUDE.key(): latitudes[is_valid],
            Dimension.LONGITUDE.key(): longitudes[is_valid],
            "first_day": pandas.Timestamp(first_day_datetime),
            "lead_day": lead_day,
            Dimension.DEPTH.key(): 0.0,
            "depth_bin": "surface",
        }
    )
