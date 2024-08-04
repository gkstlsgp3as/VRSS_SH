# -*- coding: utf-8 -*-
"""
Created on Sun May 19 15:39:52 2024

@author: Soyeon Park

Description: Extract wind speed from the Sentinel-1(C-band) satellite.

How to use: python Wind_ext_C.py --sarpath <sar file path> --sarname <sar file name> --metapath <sar metadata path> --forepath <Model forecast path> --savepath <path to save results>
Example usage: python D:\Oilspill_Total\Code_Python\Wave_spectra\KCGSA_Codes\3_Wind_ext_C.py -sp "D:/Extract_spectra/Sentinel1/Orb_Bnr_Cal_Spk_TC_LM" -sn "S1A_IW_GRDH_1SDV_20240410T094822_20240410T094851_053370_0678F1_5792_Orb_Cal_Spk_TC.tif" -mp "D:\Extract_spectra\Sentinel1" -f "D:/Extract_spectra/ECMWF_forecast_ready" -s "D:/Extract_spectra/SAR_corr"
"""

import os
import re
import argparse
from datetime import datetime
from typing import Tuple
import numpy as np
import xarray as xr
import geopandas as gpd
from osgeo import gdal
from shapely.geometry import mapping
import zipfile as zf
import xml.etree.ElementTree as ET
from typing import Dict
import time


""" common parameter file 이 만들어지면 여기 수정"""
def read_meta(sarname: str, metapath: str) -> Dict[str, float]:
    """
    Reads metadata from a ZIP file containing XML annotations for a given SAR image.
    
    Args:
        sarname (str): The name of the SAR image file.
        metapath (str): The path to the directory containing the ZIP files with metadata.
    
    Returns:
        Dict[str, float]: A dictionary containing metadata values.
    """
    
    if metapath.endswith('.zip'):
        with zf.ZipFile(metapath, mode="r") as arc:
            for xmlpath in arc.namelist():
                if 'annotation/s1a-iw-grd-vv-' in xmlpath or 'annotation/s1b-iw-grd-vv-' in xmlpath:
                    xmlstr = arc.read(xmlpath).decode('utf-8')
                    root = ET.fromstring(xmlstr)
                    break
            else:
                raise ValueError("No valid annotation file found in the ZIP archive.")
    else:
        xmlpath = metapath
        with open(xmlpath, 'r', encoding='utf-8') as file:
            xmlstr = file.read()
            root = ET.fromstring(xmlstr)
    
    meta = {
        "heading_ang": float(root.findtext('generalAnnotation/productInformation/platformHeading')),
        "start_time": root.findtext('adsHeader/startTime'),
        "prod_type": root.findtext('adsHeader/productType'),
        "pol": root.findtext('adsHeader/polarisation'),
        "band": "c-band"
    }

    geolocation_points = root.findall('.//geolocationGridPoint')
    if not geolocation_points:
        raise ValueError("No geolocationGridPoint found in the XML file.")

    first_incidence_angle = float(geolocation_points[0].find('incidenceAngle').text)
    last_incidence_angle = float(geolocation_points[-1].find('incidenceAngle').text)

    meta.update({
        "near_inc": first_incidence_angle,
        "far_inc": last_incidence_angle
    })

    if any(value is None for value in meta.values()):
        raise ValueError("Meta contains None values: {}".format(meta))

    return meta


def read_sar_and_resample(sarname: str) -> xr.Dataset:
    """
    Read SAR data and resample to 1km resolution.

    Args:
    - sarname (str): Name of the SAR file with absolute path

    Returns:
    - xr.Dataset: Processed SAR data with masking applied and resampled to 1km resolution.
    """
    dataset = gdal.Open(sarname, gdal.GA_ReadOnly)
    width, height = dataset.RasterXSize, dataset.RasterYSize
    gt = dataset.GetGeoTransform()

    sigma_band = dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
    sigma_band = np.where(sigma_band == 0, np.nan, sigma_band)
    theta_band = dataset.GetRasterBand(2).ReadAsArray().astype(np.float32)  # common parameter file 만들어지면 여기 수정
    theta_band = np.where(theta_band == 0, np.nan, theta_band)

    x1, x2 = gt[0], gt[0] + gt[1] * (width - 1)
    y1, y2 = gt[3], gt[3] + gt[5] * (height - 1)
    xr_array = xr.Dataset({
        "sigma": (["y", "x"], sigma_band),
        "theta": (["y", "x"], theta_band)
    }, coords={
        "x": np.linspace(x1, x2, width),
        "y": np.linspace(y1, y2, height)
    })
        
    del sigma_band, theta_band

    array = xr_array.rename({'x': 'lon', 'y': 'lat'})
    sigma_bs_mask = array['sigma'].where((array['sigma'] <= 1) & (array['sigma'] >= 0.0001))
    lower_bound = sigma_bs_mask.quantile(0.10, skipna=True)
    upper_bound = sigma_bs_mask.quantile(0.95, skipna=True)
    array['sigma'] = sigma_bs_mask.where((sigma_bs_mask > lower_bound) & (sigma_bs_mask < upper_bound))

    # Resample to 1km resolution
    target_resolution = 1000  # target resolution in meters
    lon_res = abs(gt[1])  # pixel size in longitude direction
    lat_res = abs(gt[5])  # pixel size in latitude direction
    resampling_factor_lon = int(target_resolution / (lon_res * 111320))
    resampling_factor_lat = int(target_resolution / (lat_res * 111320))
    array = array.coarsen(lat=resampling_factor_lat, lon=resampling_factor_lon, boundary='trim').mean()

    return array


def extract_wind_dir(forepath: str, meta: Dict[str, float]) -> Tuple[xr.DataArray, xr.DataArray, xr.Dataset, datetime]:
    """
    Extract wind direction and wind speed information from forecast data based on SAR data timestamp.

    Args:
    - forepath (str): Path to the directory containing forecast data.
    - meta (Dict[str, float]): Metadata dictionary containing SAR image metadata.

    Returns:
    - Tuple containing wind direction grid, wind speed grid, wind dataset, and target datetime.
    """

    target_time = datetime.strptime(meta['start_time'], "%Y-%m-%dT%H:%M:%S.%f")

    closest_file = None
    min_time_diff = float('inf')
    for filename in os.listdir(forepath):
        pattern = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})z")
        match = pattern.search(filename)
        if match:
            file_datetime = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)))
            time_diff = (target_time - file_datetime ).total_seconds()
            if 0 <= time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_file = filename

    fullpath = os.path.normpath(os.path.join(forepath, closest_file))
    forecast_data = xr.open_dataset(fullpath)
    print("Closest one:", fullpath)
    # forecast_data = forecast_data.rename({'latitude': 'lat', 'longitude': 'lon'})

    try:
        dir_grid = forecast_data['dir'].sel(time=target_time, method='nearest')
        spd_grid = forecast_data['speed'].sel(time=target_time, method='nearest')
    except KeyError:
        raise KeyError("Time dimension or 'dir' variable not found in the dataset.")
    
    lat_resol = abs(dir_grid.lat[1] - dir_grid.lat[0])
    lon_resol = abs(dir_grid.lon[1] - dir_grid.lon[0])
    
    lat_global = np.arange(90, -(90 + lat_resol), -lat_resol)
    lon_global = np.arange(-180, (180 + lon_resol), lon_resol)    

    dir_global = xr.DataArray(
        np.zeros((len(lat_global), len(lon_global))),
        coords = [lat_global, lon_global], dims = ['lat', 'lon'])
    dir_global.loc[dict(lat=dir_grid.lat, lon=dir_grid.lon)] = dir_grid.values
    dir_global = dir_global.assign_coords({
        'heightAboveGround': dir_grid.heightAboveGround,
        'time': dir_grid.time,
        'meanSea': dir_grid.meanSea})
    dir_global.name = "dir"     
    dir_grid = dir_global      

    return dir_grid, spd_grid, forecast_data, target_time


def cal_rel_dir(dir_grid: xr.DataArray, look: float) -> xr.DataArray:
    """
    Calculate the relative wind direction from the meteorological convention wind direction
    and the look direction.

    Args:
    - dir_grid (xr.DataArray): Wind direction in meteorological convention (degrees).
    - look (float): Look direction (degrees).

    Returns:
    - xr.DataArray: Relative wind direction (degrees) in meteorological convention.
    """
    rel_dir = (dir_grid - look) % 360
    rel_dir = xr.where(rel_dir < 0, rel_dir + 360, rel_dir)
    
    return rel_dir
    

def identical_resol(asis_grid: xr.DataArray, tobe_grid: xr.DataArray) -> xr.DataArray:
    """
    Interpolate asis_grid data to match the spatial resolution of the tobe_grid data.

    Args:
    - asis_grid (xr.DataArray): The original data grid that needs to be interpolated to a new resolution.
    - tobe_grid (xr.DataArray): The target data grid that provides the desired spatial resolution.

    Returns:
    - xr.DataArray: Interpolated data grid with the same spatial resolution as tobe_grid.
    """
    return asis_grid.interp(lat=tobe_grid.lat, lon=tobe_grid.lon, method='linear')
    # return asis_grid.interp(latitude=tobe_grid.lat, longitude=tobe_grid.lon, method='linear')


def cband_forward(v: xr.DataArray, phi: xr.DataArray, theta: xr.DataArray) -> xr.DataArray:
    """
    Compute normalized backscatter using CMOD5 model.

    Args:
    - v (xr.DataArray): Wind velocity.
    - phi (xr.DataArray): Angle between azimuth and wind direction.
    - theta (xr.DataArray): Incidence angle.

    Returns:
    - xr.DataArray: Normalized backscatter (CMOD5_N).
    """
    DTOR = 57.29577951
    THETM = 40.
    THETHR = 25.
    ZPOW = 1.6

    C = [0, -0.6878, -0.7957, 0.3380, -0.1728, 0.0000, 0.0040, 0.1103, 0.0159,
         6.7329, 2.7713, -2.2885, 0.4971, -0.7250, 0.0450, 0.0066, 0.3222, 0.0120,
         22.7000, 2.0813, 3.0000, 8.3659, -3.3428, 1.3236, 6.2437, 2.3893, 0.3249, 4.1590, 1.6930]

    FI = phi / DTOR
    CSFI = np.cos(FI)
    CS2FI = 2.00 * CSFI ** 2 - 1.00

    X = (theta - THETM) / THETHR
    XX = X ** 2

    A0 = C[1] + C[2] * X + C[3] * XX + C[4] * X * XX
    A1 = C[5] + C[6] * X
    A2 = C[7] + C[8] * X
    GAM = C[9] + C[10] * X + C[11] * XX
    S0 = C[12] + C[13] * X
    S = A2 * v

    S_vec = xr.where(S < S0, S0, S)
    A3 = 1. / (1. + np.exp(-S_vec))
    A3 = xr.where(S < S0, A3 * (S / S0) ** (S0 * (1. - A3)), A3)
    B0 = (A3 ** GAM) * 10. ** (A0 + A1 * v)

    B1 = C[15] * v * (0.5 + X - np.tanh(4. * (X + C[16] + C[17] * v)))
    B1 = C[14] * (1. + X) - B1
    B1 /= np.exp(0.34 * (v - C[18])) + 1.

    V0 = C[21] + C[22] * X + C[23] * XX
    D1 = C[24] + C[25] * X + C[26] * XX
    D2 = C[27] + C[28] * X
    V2 = (v / V0 + 1.)
    V2 = xr.where(V2 < C[19], C[19] - (C[19]-1) / C[20] + 1. / (C[20] * (C[19]-1) ** 2) * (V2-1) ** C[20], V2)
    B2 = (-D1 + D2 * V2) * np.exp(-V2)

    CMOD5_N = B0 * (1.0 + B1 * CSFI + B2 * CS2FI) ** ZPOW

    return CMOD5_N


def cband_inverse(sigma0_obs: xr.DataArray, phi: xr.DataArray, theta: xr.DataArray,
                  iterations: int = 10000, threshold: float = 0.001) -> xr.DataArray:
    """
    Retrieve wind speed by iterating the CMOD5N model until convergence with observed sigma0 values.

    Args:
    - sigma0_obs (xr.DataArray): Observed sigma0 value from SAR.
    - phi (xr.DataArray): Relative wind direction.
    - theta (xr.DataArray): Incidence angle.
    - iterations (int, optional): Number of iterations to run. Default is 1000.
    - threshold (float, optional): Error threshold for stopping iterations. Default is 0.001.

    Returns:
    - xr.DataArray: Wind speed at 10 m, neutral stratification.
    """
    V = 10.0 * xr.ones_like(sigma0_obs)
    V = V.rename("SAR wind speed")
    step = 5.0

    for _ in range(iterations):
        sigma0_calc = cband_forward(V, phi, theta)
        error = abs(sigma0_calc - sigma0_obs)
        if (error < threshold).all():
            print("The value meets threshold conditions")
            break
        adjustment = xr.where(np.isnan(error), np.nan, xr.where(sigma0_calc > sigma0_obs, -step, step))
        V += adjustment
        step /= 2
    
    return V


def xband_forward(v: xr.DataArray, phi: xr.DataArray, theta: xr.DataArray) -> xr.DataArray:
    """
    Compute normalized backscatter using XMOD2 model.

    Args:
    - v (xr.DataArray): Wind velocity.
    - phi (xr.DataArray): Angle between azimuth and wind direction.
    - theta (xr.DataArray): Incidence angle.

    Returns:
    - xr.DataArray: Normalized backscatter (XMOD2).
    """
    DTOR   = 57.29577951
    THETM  = 36.
    THETHR = 17.
    P      = 0.625
    
    # NB: 0 added as first element below, to avoid switching from 1-indexing to 0-indexing
    C = [0.0000, -1.3434, -0.7179, 0.2562,  -0.2612, 0.0312, 0.0094, 0.2527, 0.0515, 4.3308, 0.2745, -2.0974, -5.0261, -0.4141, 
          -0.0004, 0.0417, -0.0197, 0.0184, 0.0085, -0.0145, -0.0009, -0.0004, 0.0011, 
          7.4878, 0.8279, 19.6282, -14.6501, 14.4326, -0.0314, 0.1610, 0.1393, 0.6362, -0.0291]
    Y0 = C[23]
    N = C[24]
    A  = Y0-(Y0-1)/N
    B  = 1./( N* ((Y0-1.)**(N-1)) )

    #  !  ANGLES
    FI=phi/DTOR
    CSFI = np.cos(FI)
    CS2FI= 2.00 * CSFI * CSFI - 1.00

    X  = (theta - THETM) / THETHR
    XX = X**2

    #  ! B0: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    A0 =C[ 1]+C[ 2]*X+C[ 3]*XX+C[ 4]*X*XX
    A1 =C[ 5]+C[ 6]*X
    A2 =C[ 7]+C[ 8]*X
    GAM=C[ 9]+C[10]*X+C[11]*XX
    S0 =C[12]+C[13]*X
    S = A2*v
    S_vec = xr.where(S < S0, S0, S)
    A3 = 1./(1.+ np.exp(-S_vec))
    A3 = xr.where(S < S0, A3 * (S / S0) ** (S0 * (1. - A3)), A3)
    B0 = (A3 ** GAM) * 10. ** (A0 + A1 * v)
    
    #  !  B1: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    B1 = ( C[14]+C[15]*X+C[16]*XX) + (( C[17]+C[18]*X+C[19]*XX )*v) + (( C[20]+C[21]*X+C[22]*XX )*(v**2))

    #  !  B2: FUNCTION OF WIND SPEED AND INCIDENCE ANGLE
    V0 = C[25] + C[26]*X + C[27]*XX
    D1 = C[28] + C[29]*X + C[30]*XX
    D2 = C[31] + C[32]*X
    V2 = (v / V0 + 1.)
    V2 = xr.where(V2 < Y0, A + B * (V2-1)**N, V2)
    B2 = (-D1 + D2 * V2) * np.exp(-V2)

    #  !  XMOD2: COMBINE THE THREE FOURIER TERMS
    XMOD2_DLR = (B0**P) * (1 + B1*CSFI + B2*CS2FI)
    
    return XMOD2_DLR


def xband_inverse(sigma0_obs: xr.DataArray, phi: xr.DataArray, theta: xr.DataArray,
                  iterations: int = 10000, threshold: float = 0.001) -> xr.DataArray:
    """
    Retrieve wind speed by iterating the XMOD2 model until convergence with observed sigma0 values.

    Args:
    - sigma0_obs (xr.DataArray): Observed sigma0 value from SAR.
    - phi (xr.DataArray): Relative wind direction.
    - theta (xr.DataArray): Incidence angle.
    - iterations (int, optional): Number of iterations to run. Default is 1000.
    - threshold (float, optional): Error threshold for stopping iterations. Default is 0.001.

    Returns:
    - xr.DataArray: Wind speed at 10 m, neutral stratification.
    """
    V = 10.0 * xr.ones_like(sigma0_obs)
    V = V.rename("SAR wind speed")
    step = 5.0

    for i in range(iterations):
        sigma0_calc = xband_forward(V, phi, theta)
        error = abs(sigma0_calc - sigma0_obs)
        # print(f"Iteration {i}: max error = {error.max().item()}, mean error = {error.mean().item()}")
        if (error < threshold).all():
            print("The value meets threshold conditions")
            break
        adjustment = xr.where(np.isnan(error), np.nan, xr.where(sigma0_calc > sigma0_obs, -step, step))
        V += adjustment
        step /= 2

    return V


def main(sarname: str, metapath:str, forepath: str, savepath: str):
    """
    Main processing function to read SAR data, extract wind direction, interpolate data,
    and compute wind speed using the CMOD5 model.

    Args:
    - sarname (str): Name of the SAR file with its absolute path
    - forepath (str): Path to the directory containing forecast data.
    - savepath (str): Path to save the SAR wind speed result
    """
    meta = read_meta(sarname, metapath)
    sar_array = read_sar_and_resample(sarname)
    dir_grid, spd_grid, forecast_data, target_time = extract_wind_dir(forepath, meta)
    dir_grid_interp = identical_resol(dir_grid, sar_array.sigma)
    
    look_dir = ((meta['heading_ang'] + 90) + 360) % 360
    rel_dir = cal_rel_dir(dir_grid_interp, look_dir)
    
    if meta['pol'] == "VV":
        if meta['band'].lower() == "c-band":
            wind_speed = cband_inverse(sar_array.sigma, rel_dir, sar_array.theta)
        elif meta['band'].lower() == "x-band":
            wind_speed = xband_inverse(sar_array.sigma, rel_dir, sar_array.theta)
        else:
            raise ValueError(f"Unsupported band type: {meta['band']}")
    else:
        raise ValueError(f"Unsupported polarization type: {meta['pol']}")
        
    wind_speed_grid = identical_resol(wind_speed,spd_grid)

    filename = target_time.strftime('%Y_%m_%d_%H') + 'z'
    forecast_data.to_netcdf(os.path.join(forepath, filename + '.nc'))
    
    # 1. SAR wind speed 만 고해상도로 저장
    wind_speed.to_netcdf(os.path.join(savepath, filename + '_fine.nc'))

    # 2. SAR wind grid 만 forecast gird 해상도 (0.25도)로 저장
    wind_speed_grid.to_netcdf(os.path.join(savepath, filename + '_coarse.nc'))


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SAR and forecast data.")
    parser.add_argument("-sn", "--sarname", type=str, required=True, help="SAR file name with its abs path")
    parser.add_argument("-mp", "--metapath", type=str, required=True, help="Path to SAR meta data")
    parser.add_argument("-f", "--forepath", type=str, required=True, help="Path to forecast data")
    parser.add_argument("-s", "--savepath", type=str, required=True, help='Path to save the output data')

    args = parser.parse_args()

    main(args.sarname, args.metapath, args.forepath, args.savepath)



