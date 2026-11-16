# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray

from oceanbench.core.classIV_support import (
    OBSERVATION_COUNT_COLUMN,
    class4_observations_in_shared_population,
    class4_population_layers,
    create_class4_observations_dataframe,
)
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import (
    DEPTH_BIN_DISPLAY_ORDER,
    VARIABLE_DISPLAY_ORDER,
    VARIABLE_METADATA,
    Dimension,
    Variable,
)
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.neighbourhood_crps import BOX_SIZE_DIMENSION, OBSERVATION_DIMENSION, neighbourhood_crps
from oceanbench.core.references.l3_sea_surface_temperature import l3_day_observations
from oceanbench.core.resolution import TWELFTH_DEGREE_SPACING

BOX_SIZES_IN_TWELFTH_DEGREE_CELLS = [1, 3, 5, 7]


def box_sizes_on_challenger_grid(challenger: xarray.Dataset) -> list[int]:
    latitude_spacing = float(numpy.diff(challenger[Dimension.LATITUDE.key()].values).mean())
    cells_per_twelfth_degree_cell = int(numpy.rint(TWELFTH_DEGREE_SPACING / latitude_spacing))
    is_odd_fraction_of_twelfth_degree = (
        cells_per_twelfth_degree_cell >= 1
        and cells_per_twelfth_degree_cell % 2 == 1
        and numpy.isclose(cells_per_twelfth_degree_cell * latitude_spacing, TWELFTH_DEGREE_SPACING, rtol=1e-3)
    )
    if not is_odd_fraction_of_twelfth_degree:
        raise ValueError(
            "Neighbourhood CRPS needs a grid at 1/12 degree or an odd fraction of it (1/36, 1/60, ...), "
            f"got a latitude spacing of {latitude_spacing:.5f} degree."
        )
    return [box_size * cells_per_twelfth_degree_cell for box_size in BOX_SIZES_IN_TWELFTH_DEGREE_CELLS]


def _observation_column(observations_dataframe: pandas.DataFrame, column: str) -> xarray.DataArray:
    return xarray.DataArray(observations_dataframe[column].values, dims=OBSERVATION_DIMENSION)


def _observation_indexers(
    model_variable: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
) -> dict[str, xarray.DataArray]:
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    lead_day_key = Dimension.LEAD_DAY_INDEX.key()
    depth_key = Dimension.DEPTH.key()
    first_day_indices = pandas.Index(pandas.to_datetime(model_variable[first_day_key].values)).get_indexer(
        pandas.to_datetime(observations_dataframe["first_day"].values)
    )
    lead_day_indices = pandas.Index(model_variable[lead_day_key].values).get_indexer(
        observations_dataframe["lead_day"].values
    )
    time_indexers = {
        first_day_key: xarray.DataArray(first_day_indices, dims=OBSERVATION_DIMENSION),
        lead_day_key: xarray.DataArray(lead_day_indices, dims=OBSERVATION_DIMENSION),
    }
    if depth_key not in model_variable.dims:
        return time_indexers
    nearest_depth_indices = pandas.Index(model_variable[depth_key].values).get_indexer(
        observations_dataframe[depth_key].values, method="nearest"
    )
    return time_indexers | {depth_key: xarray.DataArray(nearest_depth_indices, dims=OBSERVATION_DIMENSION)}


def _crps_by_observation(
    model_variable: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
    box_sizes: list[int],
    minimum_ocean_fraction: float,
) -> pandas.DataFrame:
    crps = neighbourhood_crps(
        model_variable,
        _observation_column(observations_dataframe, Dimension.LATITUDE.key()),
        _observation_column(observations_dataframe, Dimension.LONGITUDE.key()),
        _observation_column(observations_dataframe, "observation_value"),
        box_sizes,
        minimum_ocean_fraction=minimum_ocean_fraction,
        observation_indexers=_observation_indexers(model_variable, observations_dataframe),
    )
    return (
        crps.compute()
        .to_series()
        .rename("crps")
        .reset_index()
        .merge(
            observations_dataframe[["depth_bin", "lead_day"]],
            left_on=OBSERVATION_DIMENSION,
            right_index=True,
        )
    )


def _crps_table(crps_by_observation: pandas.DataFrame, variable_key: str) -> pandas.DataFrame:
    table = crps_by_observation.groupby(["depth_bin", BOX_SIZE_DIMENSION, "lead_day"], as_index=False).agg(
        crps=("crps", "mean"),
        count=("crps", "count"),
    )
    return table.assign(variable=variable_key)


def _label(variable_key: str, depth_bin: str, box_size: int, reference_name: str | None) -> str:
    display_name, unit = VARIABLE_METADATA[variable_key]
    reference_suffix = f" vs {reference_name}" if reference_name else ""
    return (
        f"{display_name.capitalize()} ({unit}) [{variable_key}]{{{depth_bin}}} "
        f"neighbourhood {box_size}x{box_size}{reference_suffix}"
    )


def _formatted_results(
    results: pandas.DataFrame,
    lead_days_count: int,
    reference_name: str | None = None,
) -> pandas.DataFrame:
    row_keys = ["variable", "depth_bin", BOX_SIZE_DIMENSION]
    pivot_table = (
        results.pivot_table(values="crps", index=row_keys, columns="lead_day", aggfunc="first")
        .reindex(columns=range(lead_days_count))
        .reset_index()
    )
    first_lead_day_counts = results[results["lead_day"] == results["lead_day"].min()][row_keys + ["count"]]
    sorted_table = (
        pivot_table.merge(first_lead_day_counts, on=row_keys, how="left")
        .assign(
            variable_order=lambda table: table["variable"].map(VARIABLE_DISPLAY_ORDER),
            depth_order=lambda table: table["depth_bin"].map(DEPTH_BIN_DISPLAY_ORDER),
        )
        .sort_values(["variable_order", "depth_order", BOX_SIZE_DIMENSION])
    )
    labels = [
        _label(row.variable, row.depth_bin, getattr(row, BOX_SIZE_DIMENSION), reference_name)
        for row in sorted_table.itertuples(index=False)
    ]
    lead_labels = lead_day_labels(1, lead_days_count)
    result = (
        sorted_table.set_index(pandas.Index(labels))[list(range(lead_days_count)) + ["count"]]
        .rename(columns=dict(enumerate(lead_labels)) | {"count": OBSERVATION_COUNT_COLUMN})
        .rename_axis(index=None, columns=None)
    )
    return result


def crps_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    ocean_mask: xarray.DataArray,
    variables: list[Variable],
    minimum_ocean_fraction: float = 0.5,
) -> pandas.DataFrame:
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]
    box_sizes = box_sizes_on_challenger_grid(challenger)
    population_layers = class4_population_layers(ocean_mask)

    def variable_table(variable: Variable) -> pandas.DataFrame:
        observations_dataframe = create_class4_observations_dataframe(
            reference_dataset, variable.key(), variable.key(), lead_days_count
        )
        if observations_dataframe.empty:
            return pandas.DataFrame()
        shared_observations = class4_observations_in_shared_population(
            observations_dataframe.dropna(subset=["observation_value"]),
            population_layers,
        ).reset_index(drop=True)
        if shared_observations.empty:
            return pandas.DataFrame()
        return _crps_table(
            _crps_by_observation(challenger[variable.key()], shared_observations, box_sizes, minimum_ocean_fraction),
            variable.key(),
        )

    tables = [table for table in map(variable_table, variables) if not table.empty]
    if not tables:
        return pandas.DataFrame()
    return _formatted_results(pandas.concat(tables, ignore_index=True), lead_days_count)


def _surface_level(model_variable: xarray.DataArray) -> xarray.DataArray:
    if Dimension.DEPTH.key() not in model_variable.dims:
        return model_variable
    return model_variable.isel({Dimension.DEPTH.key(): 0}, drop=True)


def _crps_sums(crps_by_observation: pandas.DataFrame) -> pandas.DataFrame:
    return crps_by_observation.groupby(["depth_bin", BOX_SIZE_DIMENSION, "lead_day"], as_index=False).agg(
        crps_sum=("crps", "sum"),
        count=("crps", "count"),
    )


def crps_class4_l3_sea_surface_temperature_validation(
    challenger_dataset: xarray.Dataset,
    l3_windows: list[xarray.DataArray],
    ocean_mask: xarray.DataArray,
    minimum_ocean_fraction: float = 0.5,
) -> pandas.DataFrame:
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    temperature_key = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]
    box_sizes = box_sizes_on_challenger_grid(challenger)
    population_layers = class4_population_layers(ocean_mask)
    surface_temperature = _surface_level(challenger[temperature_key])
    first_day_datetimes = challenger[Dimension.FIRST_DAY_DATETIME.key()].values

    def day_sums(first_day_index: int, lead_day: int) -> pandas.DataFrame:
        day_observations = class4_observations_in_shared_population(
            l3_day_observations(l3_windows[first_day_index], first_day_datetimes[first_day_index], lead_day),
            population_layers,
        ).reset_index(drop=True)
        if day_observations.empty:
            return pandas.DataFrame()
        return _crps_sums(
            _crps_by_observation(surface_temperature, day_observations, box_sizes, minimum_ocean_fraction)
        )

    sums = [
        table
        for table in (
            day_sums(first_day_index, lead_day)
            for first_day_index in range(first_day_datetimes.size)
            for lead_day in range(lead_days_count)
        )
        if not table.empty
    ]
    if not sums:
        return pandas.DataFrame()
    totals = (
        pandas.concat(sums, ignore_index=True)
        .groupby(["depth_bin", BOX_SIZE_DIMENSION, "lead_day"], as_index=False)[["crps_sum", "count"]]
        .sum()
    )
    results = totals.assign(crps=totals["crps_sum"] / totals["count"], variable=temperature_key)
    return _formatted_results(results, lead_days_count, reference_name="L3S")
