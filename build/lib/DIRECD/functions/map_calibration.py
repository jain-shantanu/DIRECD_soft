import streamlit as st
import time
import numpy as np
from functions import map_calibration as mp
import astropy
import astropy.units as u
import copy
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys
import pandas as pd
import pickle
import sunpy.map
import subprocess
import time

from aiapy.calibrate import fix_observer_location, update_pointing, register
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
from reproject import reproject_interp
from scipy import ndimage, misc
from sunpy.coordinates import Helioprojective,HeliographicStonyhurst, RotatedSunFrame, transform_with_sun_center, frames
from sunpy.sun import constants as con
from sunpy.time import TimeRange, parse_time
from sunpy.net import Fido, attrs as a
from sunpy.io import write_file
from skimage.morphology import binary_closing, disk
import datetime
from sunpy.net import Fido, attrs as a

astropy.units.add_enabled_aliases({'arcsecs': astropy.units.arcsec})


def find_limb(smap_2):

    observer= smap_2.observer_coordinate
    resolution=1000
    rsun = smap_2.meta['rsun_ref']*u.m
    observer = observer.transform_to(
            HeliographicStonyhurst(obstime=observer.obstime))
    dsun = observer.radius
    if dsun <= rsun:
        raise ValueError('Observer distance must be greater than rsun')
    # Create the limb coordinate array using Heliocentric Radial
    limb_radial_distance = np.sqrt(dsun**2 - rsun**2)
    limb_hcr_rho = limb_radial_distance * rsun / dsun
    limb_hcr_z = dsun - np.sqrt(limb_radial_distance**2 - limb_hcr_rho**2)
    limb_hcr_psi = np.linspace(0, 2*np.pi, resolution+1)[:-1] << u.rad
    limb = SkyCoord(limb_hcr_rho, limb_hcr_psi, limb_hcr_z,
                    representation_type='cylindrical',
                    frame='heliocentric',
                    observer=observer, obstime=observer.obstime)
    return limb

def diff_rot(mapname, time_diff, basemap):
    
    #NOTE: to go back in time, time_diff MUST be negative
    in_time = mapname.date
    out_time = in_time + time_diff
    
    # The output frame is shifted in time for an observer at Earth (i.e., about five degrees offset in
    # heliographic longitude compared to the location of AIA in the original observation).
#     print(u.Quantity(mapname.meta['rsun_obs']), '\n"dsun_obs": ', u.Quantity(mapname.meta['dsun_obs'])) 
    out_frame = Helioprojective(observer='earth', obstime= out_time, rsun= mapname.meta['rsun_ref']*u.m)
    rot_frame = RotatedSunFrame(base=out_frame, rotated_time=in_time)
    
    # Construct a WCS object for the output map with the target RotatedSunHelioprojective frame specified
    # instead of the regular Helioprojective frame.
    out_shape = mapname.data.shape
    
    # WCS converts cdelt and cunit from arcsec to deg
    out_wcs = basemap.wcs
#     print('\nout_wcs:\n', out_wcs,'\nout_wcs type: ', type(out_wcs))
    out_wcs.coordinate_frame = rot_frame

#     For precise transformations of solar features, one should also use the context manager
#     transform_with_sun_center() to account for the translational motion of the Sun. Using the context manager,
#     the radius component stays as the solar radius as desired
#     Note: This context manager accounts only for the motion of the center of the Sun, i.e., translational motion.
#           The motion of solar features due to any rotation of the Sun about its rotational axis is not accounted for.
    # Reproject the map from the input frame to the output frame.
    with transform_with_sun_center():
        arr, _ = reproject_interp(mapname, out_wcs, out_shape)
    # Create the output map and preserve the original map’s plot settings.
    out_warp = sunpy.map.Map(arr, out_wcs)
    out_warp.plot_settings = mapname.plot_settings
    
    
    out_warp.meta["date-obs"]=mapname.meta["date-obs"]
    
##     print(out_warp.meta["date-obs"])

    if mapname.name.startswith('AIA'):
        keys_to_copy = ["waveunit", "exptime", "wavelnth", "instrume", "telescop", "rsun_obs"]
        for key in keys_to_copy:
            out_warp.meta[key] = mapname.meta[key]

    keys_to_copy = ["cdelt1", "cdelt2", "cunit1", "cunit2"]
    for key in keys_to_copy:
        out_warp.meta[key] = mapname.meta[key]

    return out_warp

def create_event_datetime(date_event, time_event):
    """Combine date and time into a datetime object."""
    event_str = f"{date_event.strftime('%Y-%m-%d')}T{time_event.strftime('%H:%M:%S')}"
    return datetime.datetime.strptime(event_str, '%Y-%m-%dT%H:%M:%S')



def download_data(wavelength, sample, start_time, end_time, save_path,data_type="base"):
    results = Fido.search(
        a.Time(start_time.strftime("%Y-%m-%dT%H:%M:%S"), end_time.strftime("%Y-%m-%dT%H:%M:%S")),
        a.Instrument('AIA'),
        a.Wavelength(wavelength * u.angstrom),
        a.Sample(sample * u.s)
    )

    if results:

        if data_type == "base":
            
            if len(results[0])==1:
                data_files = Fido.fetch(results[0], path=save_path)
                total_files = len(results[0])
            else:
                data_files = Fido.fetch(results[0][1], path=save_path)
                total_files = len(results[0][1])
            st.write('Base Files:', total_files)
        else: 
            data_files = Fido.fetch(results[0], path=save_path)
            total_files = len(results[0])
            st.write('Data Files:', total_files)
    
    else:
        st.warning('Zero Results')

    
    return data_files, total_files


def calibrate_map(sunpy_map):
    if sunpy_map.meta['lvl_num']==1:
        map_updated_pointing= update_pointing(sunpy_map)
        registered = register(map_updated_pointing)
    else:
        registered = sunpy_map
    map_2= registered/registered.exposure_time
    new_dimensions = [1024,1024] * u.pixel
    map_2 = map_2.resample(new_dimensions)
    map_time=sunpy.time.parse_time(map_2.meta['date-obs'])
    return map_2,map_time

def crop_map(smap,cenX,cenY,crop_size_x=1000,crop_size_y=1000):
    top = [cenX+(crop_size_x/2), cenY+(crop_size_y/2)]
    
    bottom = [cenX-(crop_size_x/2), cenY-(crop_size_y/2)]
    
    top_right = SkyCoord(top[0] * u.arcsec, top[1] * u.arcsec, frame=smap.coordinate_frame)
    bottom_left = SkyCoord(bottom[0] * u.arcsec, bottom[1] * u.arcsec, frame=smap.coordinate_frame)
    smap_cropped = smap.submap(bottom_left, top_right=top_right)
    return smap_cropped



def find_seeds_LBR(data_img, thresh=-0.15, perc=0.3):
    
    below_thresh_indices = np.where(data_img <= thresh)
    #below_thresh_values = data_img[data_img <= thresh]
    # Get the values of these pixels
    below_thresh_values = data_img[below_thresh_indices]
    
    # Sort the values in ascending order and get the indices that would sort the array
    sorted_indices = np.argsort(below_thresh_values)
    
    # Calculate the number of pixels to extract based on the percentage
    num_pixels_to_extract = int(np.round(perc * len(sorted_indices)))
 
    # Get the coordinates of the darkest pixels
    extracted_seeds = np.array(np.column_stack(below_thresh_indices)[sorted_indices[:num_pixels_to_extract]],dtype = np.int16)
    
    return extracted_seeds
#@jit(nopython=True)



def explore_neighbours(image_data, segmented_img, contour, current_pixel, img_shape,orientations,threshold):
    neighbours = np.array(current_pixel + orientations,dtype=np.int16)
    valid_neighbours_mask = (neighbours[:, 0] >= 0) & (neighbours[:, 0] < img_shape[0]) & \
                            (neighbours[:, 1] >= 0) & (neighbours[:, 1] < img_shape[1])
    
    valid_neighbours = neighbours[valid_neighbours_mask] 
    
    valid_mask = image_data[valid_neighbours[:, 0], valid_neighbours[:, 1]] <= threshold


    valid_neighbours_2 = valid_neighbours[valid_mask]
    
    valid_indices = valid_neighbours_2[segmented_img[valid_neighbours_2[:, 0], valid_neighbours_2[:, 1]] == 0]

    segmented_img[valid_indices[:, 0], valid_indices[:, 1]] = 150

    contour.extend(map(tuple, valid_indices.tolist()))
    
    return contour

def get_nearest_neighbour(image_data, contour, mean_seg_value):
    if not contour:  # Check if contour is empty
        return -1, 1000
    contour_arr = np.array(contour,dtype=np.uint16)
    
    contour_values = np.array(image_data[contour_arr[:, 0], contour_arr[:, 1]],dtype=np.uint16)
    dist_list = np.abs(contour_values - mean_seg_value)
    
    index = np.argmin(dist_list)
    min_dist = dist_list[index]
   
    return index, min_dist


def segment(seed_points,data_img, threshold = -0.2):
    orientations = np.array([(1,0),(0,1),(-1,0),(0,-1)]) # 4 connectiviy points
    img_shape = np.array(data_img.shape,dtype=np.uint16)
    segmented_img = np.zeros(img_shape,dtype=np.uint16)
    for k in range(len(seed_points)):
        curr_pixel = np.array(seed_points[k],dtype=np.uint16)
        if segmented_img[curr_pixel[0], curr_pixel[1]]==1: continue # pixel already explored
        contour = []
       
        seg_value = np.array(data_img[curr_pixel[0],curr_pixel[1]])
        
        while(seg_value<threshold):
            segmented_img[curr_pixel[0], curr_pixel[1]]=1
            contour = explore_neighbours(data_img,segmented_img, contour, curr_pixel, img_shape, orientations,threshold)
            nearest_neighbour_idx, dist = get_nearest_neighbour(data_img, contour, seg_value)
            if nearest_neighbour_idx==-1: break
            curr_pixel = contour[nearest_neighbour_idx]
            seg_value = data_img[curr_pixel[0]][curr_pixel[1]]
            del contour[nearest_neighbour_idx]
           
    segmented_img[segmented_img==150]=0
    return segmented_img


def delta_from_base(map_aia, base_time):
    time = map_aia.date
    time_range = TimeRange(time.value, base_time)
    return round(time_range.minutes.value,2)

def close_file(smap):
    if hasattr(smap, '_file'):  # Check if file handle exists
        smap._file.close()


## Area Calculation and smoothing functions

def rad_parallel_cut(rad):
    out=np.zeros(2*rad+1);
    for it in range (0,rad+1):
        out[it]=np.fix(np.sqrt((rad**2-(rad-it)**2)))
   
    out[rad+1:len(out)+1]=np.flipud(out[0:rad])
    return out


# Only for Surface

def coord_pixel_to_real(i,j,nanleft,dist_top,rad_circle,R,rad):
    Y=(j-nanleft[i-dist_top-1]-rad_circle[i-dist_top-1])*(R/rad);  #Axis OY is directed to the right (j - change of columns)
    Z=((i-rad-dist_top))*(R/rad);           #Axis OZ is directed to the top   (i - change of rows)
    X=np.sqrt(R**2-Y**2-Z**2);                     #Axis OX is directed to satellite A (to us)
    return X,Y,Z


def map_info(smap):
    """
    Extracts information from the Sunpy map.

    Args:
        smap (SunpyMap): Solar map object containing metadata.

    Returns:
        tuple: A tuple containing the following information:
            - rad (int): Sun radius in pixels.
            - R_km (float): Sun radius in kilometers.
            - cenX (int): X-axis reference pixel converted for Python
            - cenY (int): Y-axis reference pixel converted for Python
            - nanleft (numpy.ndarray): Distance from the left (grid start)
            - dist_top (int): Distance from the top (grid start) to the solar disk
    """

    meta = smap.meta
    try :
        meta['rsun_obs']
    except:
        meta['rsun_obs']  = meta['rsun']
    rad = int(round(meta['rsun_obs'] / meta['cdelt1']))  
    try:
        meta['rsun_ref']
    except:
        meta['rsun_ref'] = 695700000
    R_km = meta['rsun_ref'] * 10 ** (-3)  
    
    # Note that CRPIXn follows the Fortran convention of counting from 1 to N,
    #instead of from 0 to N − 1 as is done in some programming languages. Thus we put -1 to its value
    cenX = int(meta['crpix1'])-1  
    cenY = int(meta['crpix2'])-1  

    nanleft = np.array(cenX - rad_parallel_cut(rad))  
    dist_top = cenY - rad  

    return rad, R_km, cenX, cenY, nanleft, dist_top

def calculate_pixel_area(pixel_coords,smap):
      '''
  Calculate a single pixel area based on the given pixel coordinates.
    Args:
        - pixel_coords (tuple): Tuple containing the pixel coordinates (x, y) of the pixel to calculate the area for
        - smap (SunpyMap): Solar map object 
    Returns:
        - area_pixel (float): The calculated area for the specified pixel in km^2. Returns np.nan if the pixel is outside the solar disk.
      '''

  # find map parameters
      rad, R_km, cenX, cenY, nanleft, dist_top = map_info(smap)
  # calculate area of the pixel in km2
      area_pixel = calculate_integral(pixel_coords, rad, R_km, cenX, cenY, nanleft, dist_top)
  
      return area_pixel






def calculate_integral(pixel_coords, rad, R_km, cenX, cenY, nanleft, dist_top):
    """
    Calculate the area of single pixel using surface integral based on the given pixel coordinates.
    The area of surface A that has the projection on the circle S (1 pixel) is calculated using the double integral over surface:
    A=IntegralIntegral(sqrt(1+(dz/dx)**2+(dz/dy)**2))dxdy), where y=i, x=j, x^2+y^2+z^2=R^2,
    i starts from the top of disk, j starts from the left part of disk

    Args:
        - pixel_coords (tuple): Tuple containing the pixel coordinates (x, y) of the pixel to calculate the area for.
        - Output from the map_info() function:
          - rad (int): Sun radius in pixels.
          - R_km (float): Sun radius in kilometers.
          - cenX (int): X-axis reference pixel converted for Python
          - cenY (int): Y-axis reference pixel converted for Python
          - nanleft (numpy.ndarray): Distance from the left (grid start)
          - dist_top (int): Distance from the top (grid start) to the solar disk

    Returns:
        - area_pixel (float): The calculated area for the specified pixel in km^2. Returns np.nan if the pixel is outside the solar disk.
    """


    i_all, j_all = pixel_coords[1], pixel_coords[0]

    i = i_all - dist_top
    j = j_all - nanleft[rad-1]
    if j < 1:
        return np.nan

    if j_all >= cenX:
        j = rad - (j_all - cenX)
        if j < 1:
            return np.nan

    if i_all > cenY:
        i = rad - (i_all - cenY)

    sqrt_term = np.sqrt(2 * rad * j - j ** 2)

    if sqrt_term == 0:
        rad += 0.001

    R = R_km
    m = rad
    a = R / m
    dt = -R + j * a
    gm = (m - i) * a

    area_pixel = 0
    step = 1000
    dx = a / step
    x = a * (-rad + j - 0.5) + np.arange(step) * dx

    # limits of integral over Y axis
    int_up = a * (m - i + 0.5)
    int_down = a * (m - i - 0.5)

    error_list = []

    try:
        dSS = R * (np.arcsin(int_up / np.sqrt(R ** 2 - (x + dx / 2) ** 2))
                   - np.arcsin(int_down / np.sqrt(R ** 2 - (x + dx / 2) ** 2))) * dx
    except ValueError:
        dSS = np.nan + 1j
    area_pixel = np.sum(dSS)

    if np.isreal(area_pixel):
        return area_pixel
    else:
        return np.nan
    

def exp_smooth_forward(data, alpha, init=0):
    "Supplementary function for forward exponential smoothing"
    out = np.empty(data.size)
    out[0] = 0
    for i in range(1, data.size):
        out[i] = out[i-1] + alpha*(data[i]-out[i-1])
    return out

def exp_smooth_backward(data, alpha, init=0):
    "Supplementary function for backward exponential smoothing"
    out = np.empty(data.size)
    out[-1] = data[-1]
    for i in reversed(range(data.size-1)):
        out[i] = out[i+1] + alpha*(data[i]-out[i+1])
    return out

def exp_smooth(data, alpha, init=0):
    "Supplementary function for forward then backward exponential smoothing"
    return exp_smooth_backward(exp_smooth_forward(data, alpha, init), alpha, exp_smooth_forward(data, alpha, init)[-1])


def calculate_region_area_2(points_coords, smap):

  #useful map parameters for calculations from the map metadata
    rad, R_km, cenX, cenY, nanleft, dist_top = map_info(smap)

    area = 0
    for x in range(len(points_coords[0])):
    #for y in range(ylim[0], ylim[1] + 1):

      # Calculate the area for the current pixel
        pixel_coords = (points_coords[0][x], points_coords[1][x])

      # Calculate the total area of the region by summing up the individual pixel areas
        area_1 = calculate_integral(pixel_coords, rad, R_km, cenX, cenY, nanleft, dist_top)
        if ~np.isnan(area_1):
            area =  area + area_1
        #else:
            #print('Area of Pixel %d is nan',pixel_coords)
        
       

    return area

def calculate_region_area(coords, smap):
    '''
  Calculate a region area based on the list of its coordinates.
    Args:
        - coords (list): List containing coordinates (x,y) of the region area
        - smap (SunpyMap): Solar map object
    Returns:
        - area_pixel (float): The calculated area for the specified pixel in km^2. Returns np.nan if the pixel is outside the solar disk.
  '''
    #amap = np.zeros(smap.data.shape)
  #useful map parameters for calculations from the map metadata
    rad, R_km, cenX, cenY, nanleft, dist_top = map_info(smap)

    area = 0

    for x in range(len(coords)):

    # Calculate the area for the current pixel
        pixel_coords =  (coords[x][0], coords[x][1])

    # Calculate the integral for the current pixel
        integral_value = calculate_integral(pixel_coords, rad, R_km, cenX, cenY, nanleft, dist_top)
        

    # If the result is not nan, add it to the total area
        if integral_value == integral_value:
            area += integral_value
           # amap[pixel_coords[0]][pixel_coords[1]] = integral_value
   # area_map = sunpy.map.Map(amap,smap.meta)
    return area

def area_mask(smap):
# Input Binary map where 1 is dimming
    amap = np.zeros(smap.data.shape)
    rad, R_km, cenX, cenY, nanleft, dist_top = map_info(smap)
    points_coords =  np.argwhere(smap.data==1)
    for x in range(len(points_coords)):
        pixel_coords = points_coords[x]
        area_2 = calculate_integral(pixel_coords, rad, R_km, cenX, cenY, nanleft, dist_top)
        amap[pixel_coords[0]][pixel_coords[1]] = area_2
    area_map = sunpy.map.Map(amap,smap.meta)
    return area_map

def map_info_2(smap):
    """
    Extracts information from the Sunpy map.

    Args:
        smap (SunpyMap): Solar map object containing metadata.

    Returns:
        tuple: A tuple containing the following information:
            - rad (int): Sun radius in pixels.
            - R_km (float): Sun radius in kilometers.
            - cenX (int): X-axis reference pixel converted for Python
            - cenY (int): Y-axis reference pixel converted for Python
            - nanleft (numpy.ndarray): Distance from the left (grid start)
            - dist_top (int): Distance from the top (grid start) to the solar disk
    """

    meta = smap.meta
    try :
        meta['rsun_obs']
    except:
        meta['rsun_obs']  = meta['rsun']
    rad = int(round(meta['rsun_obs'] / meta['cdelt1']))  
    try:
        meta['rsun_ref']
    except:
        meta['rsun_ref'] = 695700000
    R_km = meta['rsun_ref'] * 10 ** (-3)  
    
    # Note that CRPIXn follows the Fortran convention of counting from 1 to N,
    #instead of from 0 to N − 1 as is done in some programming languages. Thus we put -1 to its value
    cenX = int(meta['crpix1'])-1  
    cenY = smap.data.shape[1]-np.floor(meta['crpix2'])-1  

    nanleft = np.array(cenX - rad_parallel_cut(rad))  
    dist_top = cenY - rad  

    return rad, R_km, cenX, cenY, nanleft, dist_top
