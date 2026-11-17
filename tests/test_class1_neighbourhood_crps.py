# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray

from oceanbench.core.class1_neighbourhood_crps import crps_class1_validation
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.references import ostia

FIRST_DAYS = numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]")
LEAD_DAYS_COUNT = 3


def _challenger(values: numpy.ndarray) -> xarray.Dataset:
    latitudes = 40 + numpy.arange(values.shape[-2]) / 12
    longitudes = -10 + numpy.arange(values.shape[-1]) / 12
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
                values,
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


def _fake_ostia_open_dataset(opened_requests: list[dict]):
    def open_dataset(**request) -> xarray.Dataset:
        opened_requests.append(request)
        days = pandas.date_range(request["start_datetime"], request["end_datetime"], freq="D")
        latitudes = numpy.arange(39.0, 42.0, 0.05)
        longitudes = numpy.arange(-11.0, -8.0, 0.05)
        temperature_in_kelvin = numpy.broadcast_to(
            283.15 + latitudes[None, :, None], (days.size, latitudes.size, longitudes.size)
        )
        sea_ice_fraction = numpy.zeros(temperature_in_kelvin.shape)
        sea_ice_fraction[:, latitudes.searchsorted(40.5) :, :] = 0.2
        dimensions = ["time", Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()]
        return xarray.Dataset(
            {
                "analysed_sst": (dimensions, temperature_in_kelvin),
                "sea_ice_fraction": (dimensions, sea_ice_fraction),
            },
            coords={"time": days, Dimension.LATITUDE.key(): latitudes, Dimension.LONGITUDE.key(): longitudes},
        )

    return open_dataset


def test_ostia_is_opened_per_forecast_window_converted_to_celsius_and_interpolated_onto_the_challenger_grid(
    monkeypatch,
):
    opened_requests: list[dict] = []
    monkeypatch.setattr(ostia.copernicusmarine, "open_dataset", _fake_ostia_open_dataset(opened_requests))
    challenger = _challenger(numpy.zeros((2, LEAD_DAYS_COUNT, 2, 6, 5)))

    reference = ostia.ostia_sea_surface_temperature_dataset(challenger)

    assert [request["start_datetime"] for request in opened_requests] == ["2024-01-03T00:00:00", "2024-01-10T00:00:00"]
    assert opened_requests[0]["end_datetime"] == "2024-01-05T00:00:00"
    assert opened_requests[0]["dataset_id"] == ostia.OSTIA_REPROCESSED_DATASET_IDENTIFIER
    temperature = reference[Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()]
    assert temperature.dims == (
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    )
    ice_free_rows = challenger[Dimension.LATITUDE.key()].values < 40.45
    ice_free_temperature = temperature.isel({Dimension.LATITUDE.key(): ice_free_rows})
    expected_celsius = 10 + ice_free_temperature[Dimension.LATITUDE.key()]
    numpy.testing.assert_allclose(
        ice_free_temperature.values,
        expected_celsius.broadcast_like(ice_free_temperature).values,
        atol=1e-4,
    )
    assert temperature.isel({Dimension.LATITUDE.key(): ~ice_free_rows}).isnull().all()


def test_class1_table_has_one_area_weighted_row_per_box_size():
    challenger_values = numpy.full((2, LEAD_DAYS_COUNT, 2, 12, 12), 15.0)
    challenger_values[..., 0, 0] = numpy.nan
    challenger = _challenger(challenger_values)
    reference = challenger.isel({Dimension.DEPTH.key(): 0}, drop=True) + 1.0

    table = crps_class1_validation(challenger, reference, reference_name="OSTIA")

    assert list(table.columns) == ["Lead day 1", "Lead day 2", "Lead day 3"]
    assert table.index[3].endswith("neighbourhood 7x7 vs OSTIA")
    numpy.testing.assert_allclose(table.values, 1.0)


def test_ostia_accepts_a_challenger_whose_coordinates_are_named_only_by_standard_name(monkeypatch):
    monkeypatch.setattr(ostia.copernicusmarine, "open_dataset", _fake_ostia_open_dataset([]))
    challenger = _challenger(numpy.zeros((2, LEAD_DAYS_COUNT, 2, 6, 5))).rename(
        {Dimension.LATITUDE.key(): "lat", Dimension.LONGITUDE.key(): "lon"}
    )
    challenger["lat"].attrs["standard_name"] = Dimension.LATITUDE.key()
    challenger["lon"].attrs["standard_name"] = Dimension.LONGITUDE.key()

    reference = ostia.ostia_sea_surface_temperature_dataset(challenger)

    assert reference.sizes[Dimension.LATITUDE.key()] == 6
    assert reference.sizes[Dimension.LONGITUDE.key()] == 5
