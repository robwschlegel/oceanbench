# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray

from oceanbench.core.class4_neighbourhood_crps import crps_class4_l3_sea_surface_temperature_validation
from oceanbench.core.classIV_support import OBSERVATION_COUNT_COLUMN
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.references import l3_sea_surface_temperature

FIRST_DAYS = numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]")
LEAD_DAYS_COUNT = 2
L3_LATITUDES = 40.05 + 0.1 * numpy.arange(8)
L3_LONGITUDES = -9.95 + 0.1 * numpy.arange(8)


def _l3_dataset(days: pandas.DatetimeIndex) -> xarray.Dataset:
    shape = (days.size, L3_LATITUDES.size, L3_LONGITUDES.size)
    temperature_in_kelvin = numpy.full(shape, 288.15)
    quality_level = numpy.full(shape, 5.0)
    quality_level[:, 0, :] = 3
    temperature_in_kelvin[:, 1, :] = numpy.nan
    dimensions = ["time", Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()]
    return xarray.Dataset(
        {
            "sea_surface_temperature": (dimensions, temperature_in_kelvin),
            "quality_level": (dimensions, quality_level),
        },
        coords={"time": days, Dimension.LATITUDE.key(): L3_LATITUDES, Dimension.LONGITUDE.key(): L3_LONGITUDES},
    )


def _fake_open_dataset(opened_requests: list[dict]):
    def open_dataset(**request) -> xarray.Dataset:
        opened_requests.append(request)
        return _l3_dataset(pandas.date_range(request["start_datetime"], request["end_datetime"], freq="D"))

    return open_dataset


def _challenger(temperature: float) -> xarray.Dataset:
    latitudes = 39.75 + numpy.arange(18) / 12
    longitudes = -10.25 + numpy.arange(18) / 12
    shape = (FIRST_DAYS.size, LEAD_DAYS_COUNT, 2, latitudes.size, longitudes.size)
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
                numpy.full(shape, temperature),
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): FIRST_DAYS,
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(LEAD_DAYS_COUNT),
            Dimension.DEPTH.key(): numpy.array([0.494, 1.541]),
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def _all_ocean_mask(challenger: xarray.Dataset) -> xarray.DataArray:
    return xarray.DataArray(
        numpy.ones(
            (2, challenger.sizes[Dimension.LATITUDE.key()], challenger.sizes[Dimension.LONGITUDE.key()]), dtype=bool
        ),
        dims=[Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            Dimension.DEPTH.key(): [0.494, 1.541],
            Dimension.LATITUDE.key(): challenger[Dimension.LATITUDE.key()].values,
            Dimension.LONGITUDE.key(): challenger[Dimension.LONGITUDE.key()].values,
        },
    )


def test_l3_day_observations_keep_good_quality_pixels_in_celsius(monkeypatch):
    opened_requests: list[dict] = []
    monkeypatch.setattr(
        l3_sea_surface_temperature.copernicusmarine, "open_dataset", _fake_open_dataset(opened_requests)
    )

    windows = l3_sea_surface_temperature.l3_sea_surface_temperature_windows(_challenger(15.0))
    observations = l3_sea_surface_temperature.l3_day_observations(windows[1], FIRST_DAYS[1], lead_day=1)

    assert [request["start_datetime"] for request in opened_requests] == ["2024-01-03T00:00:00", "2024-01-10T00:00:00"]
    assert opened_requests[0]["dataset_id"] == l3_sea_surface_temperature.L3_SEA_SURFACE_TEMPERATURE_DATASET_IDENTIFIER
    assert len(observations) == 6 * L3_LONGITUDES.size
    numpy.testing.assert_allclose(observations["observation_value"], 15.0)
    assert observations[Dimension.LATITUDE.key()].min() > L3_LATITUDES[1]
    assert set(observations["lead_day"]) == {1}
    assert set(observations["first_day"]) == {pandas.Timestamp("2024-01-10")}


def test_l3_crps_pools_every_start_and_reports_the_constant_error(monkeypatch):
    monkeypatch.setattr(l3_sea_surface_temperature.copernicusmarine, "open_dataset", _fake_open_dataset([]))
    challenger = _challenger(16.5)

    table = crps_class4_l3_sea_surface_temperature_validation(
        challenger,
        l3_sea_surface_temperature.l3_sea_surface_temperature_windows(challenger),
        _all_ocean_mask(challenger),
    )

    assert len(table) == 4
    assert table.index[0].endswith("{surface} neighbourhood 1x1 vs L3S")
    numpy.testing.assert_allclose(table[["Lead day 1", "Lead day 2"]].values, 1.5)
    assert (table[OBSERVATION_COUNT_COLUMN] > 0).all()
    assert table[OBSERVATION_COUNT_COLUMN].nunique() == 1
