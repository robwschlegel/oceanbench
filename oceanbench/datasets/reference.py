# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# flake8: noqa

"""
This module exposes the reference datasets used in OceanBench for OceanBench challenger to explore.
"""

import pandas
import xarray
from oceanbench.core.references import glorys
from oceanbench.core.remote_http import open_remote_multizarr
from oceanbench.core.references import glo12
from oceanbench.core.references import ibi36


def ibi36_reference(
    start_date: str,
    end_date: str,
    username: str | None = None,
    password: str | None = None,
    minimum_longitude: float | None = None,
    maximum_longitude: float | None = None,
    minimum_latitude: float | None = None,
    maximum_latitude: float | None = None,
    minimum_depth: float | None = None,
    maximum_depth: float | None = None,
) -> xarray.Dataset:
    """
    Open Copernicus Marine IBI36 fields with their native ocean mask.

    Parameters
    ----------
    start_date, end_date : str
        Inclusive calendar dates. Source timestamps are at midnight UTC.
    username, password : str, optional
        Copernicus Marine credentials. If omitted, the Toolbox uses its
        configured environment or credentials file and may prompt.
    minimum_longitude, maximum_longitude : float, optional
        Optional limits for the longitude subset.
    minimum_latitude, maximum_latitude : float, optional
        Optional limits for the latitude subset.
    minimum_depth, maximum_depth : float, optional
        Optional positive-down depth limits in metres.

    Returns
    -------
    Dataset
        Lazy potential temperature, practical salinity, eastward and northward
        currents, and sea level, with land and below-bottom cells masked.
        Close the dataset after use.

    Uses version ``202511`` of ``IBI_MULTIYEAR_PHY_005_002``. No data files are
    written, and credentials are not saved by this function.

    >>> with ibi36_reference(  # doctest: +SKIP
    ...     "2023-12-11", "2023-12-11",
    ...     minimum_longitude=-10, maximum_longitude=-9.95,
    ...     minimum_latitude=43, maximum_latitude=43.05,
    ...     maximum_depth=10,
    ... ) as reference:
    ...     subset = reference.isel(time=0, depth=slice(0, 3))
    """
    return ibi36.ibi36_reference(
        start_date,
        end_date,
        username,
        password,
        minimum_longitude,
        maximum_longitude,
        minimum_latitude,
        maximum_latitude,
        minimum_depth,
        maximum_depth,
    )


def glorys_reanalysis() -> xarray.Dataset:
    """
    Open GLORYS reanalysis with as an `xarray.Dataset`.

    Returns
    -------
    Dataset
        The Dataset containing GLORYS reanalysis.

    Examples
    --------
    >>> glorys_reanalysis() # doctest: +SKIP
    <xarray.Dataset> Size: 5TB
    Dimensions:    (depth: 50, latitude: 2041, longitude: 4320, time: 366)
    Coordinates:
      * depth      (depth) float32 200B 0.494 1.541 2.646 ... 5.275e+03 5.728e+03
      * latitude   (latitude) float32 8kB -80.0 -79.92 -79.83 ... 89.83 89.92 90.0
      * longitude  (longitude) float32 17kB -180.0 -179.9 -179.8 ... 179.8 179.9
      * time       (time) datetime64[ns] 3kB 2024-01-01 2024-01-02 ... 2024-12-31
    Data variables:
        thetao     (time, depth, latitude, longitude) float64 1TB dask.array<chunksize=(28, 1, 512, 2048), meta=np.ndarray>
        so         (time, depth, latitude, longitude) float64 1TB dask.array<chunksize=(28, 1, 512, 2048), meta=np.ndarray>
        uo         (time, depth, latitude, longitude) float64 1TB dask.array<chunksize=(28, 1, 512, 2048), meta=np.ndarray>
        vo         (time, depth, latitude, longitude) float64 1TB dask.array<chunksize=(28, 1, 512, 2048), meta=np.ndarray>
        zos        (time, latitude, longitude) float64 26GB dask.array<chunksize=(28, 512, 2048), meta=np.ndarray>
    Attributes:
        ...
    """

    return glorys.glorys_reanalysis()


def glorys_reanalysis_1_degree() -> xarray.Dataset:
    """
    Open the historical 1 degree GLORYS reanalysis as an `xarray.Dataset`.

    This dataset is intended to support training on the 1 degree OceanBench track.
    It exposes the historical GLORYS reanalysis published at 1 degree resolution
    in the OceanBench public EDITO bucket.

    Returns
    -------
    Dataset
        The Dataset containing the historical 1 degree GLORYS reanalysis.

    Examples
    --------
    >>> glorys_reanalysis_1_degree() # doctest: +SKIP
    <xarray.Dataset> Size: 970GB
    Dimensions:                          (time: 9861, depth: 50, latitude: 170,
                                          longitude: 360)
    Coordinates:
      * depth                            (depth) float32 200B 0.494 ... 5.728e+03
      * latitude                         (latitude) float64 1kB -79.5 -78.5 ... 89.5
      * longitude                        (longitude) float64 3kB -179.5 ... 179.5
      * time                             (time) datetime64[ns] 79kB 1993-01-01 .....
    Data variables:
        eastward_sea_water_velocity      (time, depth, latitude, longitude) float64 241GB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
        northward_sea_water_velocity     (time, depth, latitude, longitude) float64 241GB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
        sea_surface_height_above_geoid   (time, latitude, longitude) float64 5GB dask.array<chunksize=(1, 170, 360), meta=np.ndarray>
        sea_water_potential_temperature  (time, depth, latitude, longitude) float64 241GB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
        sea_water_salinity               (time, depth, latitude, longitude) float64 241GB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
    Attributes:
        ...
    """

    monthly_dates = pandas.date_range("1993-01-01", "2019-12-01", freq="MS")
    dataset_paths = [
        f"https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/glorys_1degree_1993_2019/{monthly_date.strftime('%Y%m')}.zarr"
        for monthly_date in monthly_dates
    ]

    return open_remote_multizarr(
        dataset_paths,
        parallel=False,
    )


def glo12_analysis() -> xarray.Dataset:
    """
    Open GLO12 analysis with as an `xarray.Dataset`.

    Returns
    -------
    Dataset
        The Dataset containing GLO12 analysis.

    Examples
    --------
    >>> glo12_analysis() # doctest: +SKIP
    <xarray.Dataset> Size: 3TB
    Dimensions:    (depth: 50, latitude: 2041, longitude: 4320, time: 366)
    Coordinates:
      * depth      (depth) float32 200B 0.494 1.541 2.646 ... 5.275e+03 5.728e+03
      * latitude   (latitude) float32 8kB -80.0 -79.92 -79.83 ... 89.83 89.92 90.0
      * longitude  (longitude) float32 17kB -180.0 -179.9 -179.8 ... 179.8 179.9
      * time       (time) datetime64[ns] 3kB 2024-01-01 2024-01-02 ... 2024-12-31
    Data variables:
        thetao     (time, depth, latitude, longitude) float32 645GB dask.array<chunksize=(21, 1, 512, 2048), meta=np.ndarray>
        so         (time, depth, latitude, longitude) float32 645GB dask.array<chunksize=(21, 1, 512, 2048), meta=np.ndarray>
        uo         (time, depth, latitude, longitude) float32 645GB dask.array<chunksize=(21, 1, 512, 2048), meta=np.ndarray>
        vo         (time, depth, latitude, longitude) float32 645GB dask.array<chunksize=(21, 1, 512, 2048), meta=np.ndarray>
        zos        (time, latitude, longitude) float32 13GB dask.array<chunksize=(21, 1024, 2048), meta=np.ndarray>
    Attributes:
        ...
    """
    return glo12.glo12_analysis()
