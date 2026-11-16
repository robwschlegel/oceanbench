# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the functions to compute metrics.
"""

import xarray

from pandas import DataFrame

from oceanbench.core import metrics
from oceanbench.regions import RegionLike


def rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of variables compared to observations.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_variables_compared_to_observations(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def rmsd_of_variables_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of variables compared to GLORYS reanalysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_variables_compared_to_glorys_reanalysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLORYS reanalysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLORYS reanalysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the deviation of Lagrangian trajectories compared to GLORYS reanalysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def rmsd_of_variables_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of variables compared to GLO12 analysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_variables_compared_to_glo12_analysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLO12 analysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """

    return metrics.rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLO12 analysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
) -> DataFrame:
    """
    Compute the deviation of Lagrangian trajectories compared to GLO12 analysis.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
        challenger_dataset=challenger_dataset,
        region=region,
    )


def neighbourhood_crps_of_temperature_compared_to_observations(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
    minimum_ocean_fraction: float = 0.5,
) -> DataFrame:
    """
    Compute the neighbourhood Continuous Ranked Probability Score (CRPS) of temperature compared to observations.

    The challenger values in a box of grid cells around each in situ temperature observation (Argo profiles and
    drifters) form a pseudo-ensemble, scored with the CRPS (HiRA method, Mittermaier 2014). Boxes are 1, 3, 5 and 7
    cells of 1/12 degree wide, scaled to the challenger's grid, so models of different resolutions are scored over
    the same area. Only grids at 1/12 degree or an odd fraction of it are supported. Land cells are left out of a box;
    an observation is scored only if its nearest cell is ocean and every box holds at least
    ``minimum_ocean_fraction`` ocean cells, so every box size is scored on the same observations. A box of one cell
    gives the absolute error at the nearest grid cell. Lower is better.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.
    minimum_ocean_fraction : float, optional
        Minimum share of ocean cells in every box for an observation to be scored, in (0, 1]. 1 keeps only
        observations whose boxes are entirely ocean.

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.neighbourhood_crps_of_temperature_compared_to_observations(
        challenger_dataset=challenger_dataset,
        region=region,
        minimum_ocean_fraction=minimum_ocean_fraction,
    )


def neighbourhood_crps_of_sea_surface_temperature_compared_to_ostia(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = "global",
    minimum_ocean_fraction: float = 0.5,
) -> DataFrame:
    """
    Compute the neighbourhood Continuous Ranked Probability Score (CRPS) of sea surface temperature compared to OSTIA.

    At every ocean grid cell, the challenger's top-level temperature in a box of grid cells around it forms a
    pseudo-ensemble, scored with the CRPS against the OSTIA reprocessed sea surface temperature analysis
    (METOFFICE-GLO-SST-L4-REP-OBS-SST) interpolated onto the challenger grid (HiRA method, Mittermaier 2014). Boxes
    are 1, 3, 5 and 7 cells of 1/12 degree wide, scaled to the challenger's grid. Cells with any sea ice in OSTIA are
    left out. Land cells are left out of a box; a cell is scored only if it is ocean and every box holds at least
    ``minimum_ocean_fraction`` ocean cells. Scores are area-weighted means per lead day. Lower is better.

    Parameters
    ----------
    challenger_dataset : xarray.Dataset
        The challenger dataset.
    region : str, optional
        The OceanBench region to evaluate on.
    minimum_ocean_fraction : float, optional
        Minimum share of ocean cells in every box for a cell to be scored, in (0, 1].

    Returns
    -------
    DataFrame
        The DataFrame containing the scores.
    """
    return metrics.neighbourhood_crps_of_sea_surface_temperature_compared_to_ostia(
        challenger_dataset=challenger_dataset,
        region=region,
        minimum_ocean_fraction=minimum_ocean_fraction,
    )
