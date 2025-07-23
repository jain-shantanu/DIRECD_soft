from functions import map_calibration as mp
import streamlit as st
from datetime import datetime, timedelta
import os
import sunpy.map
from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.coordinates import Helioprojective, frames
import numpy as np
from scipy import ndimage
from skimage.morphology import binary_closing
import matplotlib.pyplot as plt
from moviepy.editor import *
from pathlib import Path
import pandas as pd
import pickle
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import gc
current_dir = os.getcwd()
logo_folder = os.path.join(current_dir, 'logo', 'DIRECD_logo.jpg')

def main_text():
    st.subheader('About this application:')
    st.markdown("""
                   _DIRECD_  is an open-source software package that can be used to
                   reconstruct estimate Coronal Mass Ejections (CMEs) direction from coronal dimming information.
                   The tool implements the dimming detection using region growing algorithm on SDO/AIA images.
                """)
    right, left = st.columns((1, 1))
    # right.markdown("""
    #                **Github**: Find the latest version here
    #                            [![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AthKouloumvakos/PyThea) \n
    #                **Documentation Page**: https://www.pythea.org/
    #                """ +
    #                f"""
    #                **Version**: {version} (latest release [![Version](https://img.shields.io/github/v/release/AthKouloumvakos/PyThea)](https://github.com/AthKouloumvakos/PyThea/releases))
    #                """)
    left.image(logo_folder)
    st.markdown("""
                **Citation**: Please cite the following paper (https://doi.org/10.1051/0004-6361/202347927)
                """)
    
    st.markdown('---')


st.set_page_config(page_title='Dimming Detection',
                   page_icon=logo_folder)
st.header("DIRECD: Dimming Inferred Estimation of CME Direction")

st.markdown('---')




def init_session_state():
    
    if 'data_downloaded' not in st.session_state:
        st.session_state.data_downloaded = False
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {
            'base_map_cropped': None,
            'listdir': [],
            'log_diff_folder': None,
            'cenX': None,
            'cenY': None,
            'base_time': None,
            'diff_rot_folder': None
        }
    if 'date_event' not in st.session_state:
        st.session_state.date_event = '2013-05-17'
    if 'time_event' not in st.session_state:
        st.session_state.time_event = '20:44:00'
    
    if 'wavelength_val' not in st.session_state:
        st.session_state.wavelength_val = 211
    if 'cadence_val' not in st.session_state:
        st.session_state.cadence_val = 60
    if 'prev_time_range' not in st.session_state:
        st.session_state.prev_time_range = 2
    if 'prev_lbr_thresh' not in st.session_state:
        st.session_state.prev_lbr_thresh = -0.15
    if 'prev_lon' not in st.session_state:
        st.session_state.prev_lon = 0
    if 'prev_lon_dir' not in st.session_state:
        st.session_state.prev_lon_dir = 'West'
    if 'prev_lat' not in st.session_state:
        st.session_state.prev_lat = 0
    if 'prev_lat_dir' not in st.session_state:
        st.session_state.prev_lat_dir = 'North'
    if 'first_run' not in st.session_state:
        st.session_state.first_run = True

    if 'save_plots_checkbox' not in st.session_state:
        st.session_state.save_plots_checkbox = False
    
   

def detect_parameter_changes(date_event, time_event, wavelength, cadence, lbr_thresh, time_range, lon_val,lon_dir,lat_val,lat_dir,save_plots_checkbox):
    # Initialize change tracking
    changes = {
        'any_changed': False,
        'date_event' : False,
        'time_event': False,
        'wavelength': False,
        'cadence': False,
        'lbr_thresh': False,
        'time_range': False,
        'lon_val': False,
        'lon_dir': False,
        'lat_val': False,
        'lat_dir': False,
        'save_plots_checkbox': False,
        'changed_params': []
    }
    
    # Check each parameter
    if st.session_state.date_event != date_event:
        st.session_state.date_event = date_event
        changes['date_event'] = True
        changes['changed_params'].append('date_event')
    if st.session_state.time_event != time_event:
        st.session_state.time_event = time_event
        changes['time_event'] = True
        changes['changed_params'].append('time_event')
    if st.session_state.wavelength_val != wavelength:
        st.session_state.wavelength_val = wavelength
        changes['wavelength'] = True
        changes['changed_params'].append('wavelength')
    
    if st.session_state.cadence_val != cadence:
        st.session_state.cadence_val = cadence
        changes['cadence'] = True
        changes['changed_params'].append('cadence')
    
    if st.session_state.prev_lbr_thresh != lbr_thresh:
        st.session_state.prev_lbr_thresh = lbr_thresh
        
        changes['lbr_thresh'] = True
        changes['changed_params'].append('lbr_thresh')
       
    if st.session_state.prev_time_range != time_range:
        st.session_state.prev_time_range = time_range
        changes['time_range'] = True
        changes['changed_params'].append('time_range')
    if st.session_state.prev_lat != lat_val:
        st.session_state.prev_lat = lat_val
        changes['lat_val'] = True
        changes['changed_params'].append('Latitude Value')
    if st.session_state.prev_lat_dir != lat_dir:
        st.session_state.prev_lat_dir = lat_dir
        changes['lat_dir'] = True
        changes['changed_params'].append('Latitude Direction')
    if st.session_state.prev_lon != lon_val:
        st.session_state.prev_lon = lon_val
        changes['lon_val'] = True
        changes['changed_params'].append('Longitude Value')
    if st.session_state.prev_lon_dir != lon_dir:
        st.session_state.prev_lon_dir = lon_dir
        changes['lon_dir'] = True
        changes['changed_params'].append('Longitude Direction')
    if st.session_state.save_plots_checkbox != save_plots_checkbox:
        st.session_state.save_plots_checkbox = save_plots_checkbox
        changes['save_plots_checkbox'] = True
        changes['changed_params'].append('save_plots_checkbox')
    
    
    
    # Set overall change flag
    changes['any_changed'] = len(changes['changed_params']) > 0
    
    return changes




init_session_state()


list_events =st.sidebar.selectbox('Event', ('New Event','26/11/2011T06:09'), index=0,
    placeholder="list_events")
st.session_state.list_events = list_events

default_date = "today"
default_time = '20:44'
default_flare_lat = 0
default_index_lat = 0
default_flare_lon = 0
default_index_lon = 0
default_threshold = -0.19

if st.session_state.list_events == '26/11/2011T06:09':
    default_date = datetime.strptime('26 11 2011', '%d %m %Y')
    default_time = datetime.strptime('06:09', '%H:%M')
    default_flare_lat = 11
    default_flare_lon = 47
    default_index_lat = 0
    default_index_lon = 0
    default_threshold = -0.15



form = st.sidebar.form("Event_selection")
date_col, time_col = form.columns([5, 4])
date_str_min = "01/01/2009"
parsed_date = datetime.strptime(date_str_min, "%d/%m/%Y").date()
with date_col:
    date_event = st.date_input("Date",value=default_date,min_value=parsed_date)  

with time_col:
    time_event = st.time_input("Time",value=default_time,step=60) 


time_range = form.select_slider(
    "Select the time range of detection (minutes)",
    options=[
        3,
        10,
        30,
        60,
        90,
        120,
        150,
        180,
     
    ], key="time_range"
)


wavelength_col, sample_col = form.columns([2, 2])

with wavelength_col:
    wavelength = st.selectbox(
    "Wavelength (A)",
    (193, 211),
    index=1,
    placeholder="Wavelength",
    
)

with sample_col:
   cadence = st.selectbox(
    "Cadence (seconds)",
    (12, 24,36,48,60,90,120),
    index=4,
    placeholder="Cadence (seconds)",
    
)
lat_value, lat_direction = form.columns([2, 2])

with lat_value:
    lat_val = st.slider("Flare Source Lat", 0, 60, default_flare_lat)
with lat_direction:
    lat_dir = st.selectbox(" ",("North", "South"),index = default_index_lat,
)

    
lon_value, lon_direction = form.columns([2, 2])

with lon_value:
    lon_val = st.slider("Flare Source Lon", 0, 60, default_flare_lon)
with lon_direction:
    lon_dir = st.selectbox(" ",("West", "East"),index=default_index_lon,
)

lbr_thresh = form.slider(
    "LBR Threshold",
    min_value=-0.19,
    max_value=-0.11,
    value=default_threshold,
    step=0.04,
    key="lbr_thresh_slider"
    
)

with form.expander("Save Options"):

    download_fits = st.checkbox('Overwrite Raw fits (from VSO)',label_visibility="visible")
    overwrite_files_calibration = st.checkbox('Overwrite Calibrated Data',label_visibility="visible")
    overwrite_files_lbr = st.checkbox('Overwrite LBR files',label_visibility="visible")
    
   
with form.expander("Plot Options"):

    save_plots_checkbox = st.checkbox('Save All Dimming Detection plots',label_visibility="visible")
    save_plots_checkbox_all = st.checkbox('Save all final plots',label_visibility="visible")
    
    

submit_1, submit_2 = form.columns([2, 2])


with submit_1:
    submit = st.form_submit_button("Submit")
# with submit_2:
#     submit_3 = st.form_submit_button("Submit_2")
if not submit:
    main_text()


# Create datetime objects
event_dt = mp.create_event_datetime(date_event, time_event)
timestamp_end = event_dt + timedelta(minutes=(time_range-1))
timestamp_start = event_dt - timedelta(minutes=30)
timestamp_sample_start = timestamp_start + timedelta(seconds=(cadence-1))

safe_event = event_dt.strftime("%Y-%m-%dT%H:%M:%S").replace(":", "-")
current_dir = os.getcwd()




should_download = False
should_calibrate=False
should_detect = False
should_LBR = False


# Move the initialization inside the submit block



if submit:

    changes = detect_parameter_changes(date_event, time_event,wavelength, cadence, lbr_thresh, time_range, lon_val, lon_dir, lat_val, lat_dir, save_plots_checkbox)
    if save_plots_checkbox_all:
        save_path_plots = os.path.join(current_dir, 'Events', safe_event, 'plots',str(wavelength),str(cadence))
        os.makedirs(save_path_plots, exist_ok=True) 

 
    if changes['any_changed']:
        st.write(f"Parameters changed: {', '.join(changes['changed_params'])}")
        
        # If wavelength, cadence or time range changed
        if any([changes['date_event'], changes['time_event'],changes['wavelength'], changes['cadence'], changes['time_range']]):
            should_download = True
            should_calibrate = True
            should_LBR = True
            should_detect = True
            
        # If location parameters changed
        elif any([changes['lat_val'], changes['lon_val'], changes['lat_dir'], changes['lon_dir']]):
            should_calibrate = True  # Need to recrop with new coordinates
            should_LBR = True
            should_detect = True
            
        # If only threshold changed
        elif any([changes['lbr_thresh'], changes['save_plots_checkbox']]):
            should_detect = True  # Just rerun detection with new threshold
            
    else:
        if st.session_state.first_run:
        # No changes - rerun everything
            should_download = True
            should_calibrate = True
            should_LBR = True
            should_detect = True
            st.session_state.first_run = False
        else:
            should_download = False
            should_calibrate = False
            should_LBR = False
            should_detect = False
            st.write('Parameters unchanged. Please refresh the page or change parameters')
            st.stop()

# Only process if submit was clicked
if submit:
    if should_download:
        st.header('Download Data')
        st.markdown('Here we download the required data for our dimming detection algorithm. '
                'If you already have raw/calibrated fits files, add them to necessary folders and the downloading/calibrating'
                'of files will be skipped')
        
        
        save_path_fits = os.path.join(current_dir, 'Events', safe_event, 'fits',str(wavelength),str(cadence))
        os.makedirs(save_path_fits, exist_ok=True) 
    
        if len(os.listdir(save_path_fits))==0 or download_fits:
            spinner_placeholder = st.empty()
            with spinner_placeholder:
                with st.spinner("Downloading Base Data...", show_time=True):
                    try:
                        # Download base map
                        base_data, _ = mp.download_data(
                            wavelength, cadence, 
                            timestamp_start, timestamp_sample_start, save_path_fits,
                            data_type="base"
                        )
                        
                        
                        
                    except Exception as e:
                        st.error(f"An error occurred during download: {str(e)}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.link_button("JSOC", 'http://jsoc.stanford.edu/AIA/AIA_lev1.html')
                        with col2:
                            st.link_button("MEDOC", 'https://idoc-medoc.ias.u-psud.fr/sitools/client-user/index.html?project=Medoc-Solar-Portal')
                        st.error('Base Map could not be downloaded. Cannot download Base Map from VSO, please download files manually.')
                        st.stop()
                    
            if not base_data.errors:
                            st.success(f"Base Map Downloaded: {timestamp_start}")
           
                            
            
            with spinner_placeholder:
                with st.spinner("Downloading Main Data...", show_time=True):
                    try:            
                        # Download main data
                        main_data, total_files = mp.download_data(
                            wavelength, cadence,
                            event_dt, timestamp_end, save_path_fits,
                            data_type="main"
                        )
                        

                            
                    except Exception as e:
                        st.error(f"An error occurred during download: {str(e)}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.link_button("JSOC", 'http://jsoc.stanford.edu/AIA/AIA_lev1.html')
                        with col2:
                            st.link_button("MEDOC", 'https://idoc-medoc.ias.u-psud.fr/sitools/client-user/index.html?project=Medoc-Solar-Portal')
                        st.error('Data could not be downloaded. Cannot download Data from VSO, please download files manually.')
                        st.stop()
            if not main_data.errors:
                st.success(f"Downloaded {total_files} files to: {save_path_fits}")
                st.session_state.data_downloaded = True    
        else:
            st.warning('Data Already downloaded, Skipping this step')
            st.session_state.data_downloaded = True
            should_download = False
                
    if should_calibrate:
        st.header('Calibrate Data')
        st.markdown('Here we calibrate Base Map and other fits files using sunpy functions. If you already have calibrated'
                    'data, you can put them in necessary folders and it will skip the calibration. However, it is highly recommended'
                    'to recalibrate the data for proper working of dimming detection')
        diff_rot_folder = os.path.join(current_dir, 'Events', safe_event, 'Calibrated',str(wavelength),str(cadence))
        os.makedirs(diff_rot_folder, exist_ok=True)
        log_diff_folder = os.path.join(current_dir, 'Events', safe_event, 'LBR',str(wavelength),str(cadence))
        os.makedirs(log_diff_folder, exist_ok=True)


        st.write('Calibrate Base_map')
        with st.spinner("Calibrating Base Map...",show_time=True):
            basemap_folder = os.path.join(current_dir, 'Events', safe_event, 'basemap',str(wavelength),str(cadence))
            os.makedirs(basemap_folder, exist_ok=True)
            
            listdir = sorted(os.listdir(save_path_fits)) #show elements in the folder
            number= len(listdir)

            basemap_file=os.path.join(save_path_fits, listdir[0])
            base_raw=sunpy.map.Map(basemap_file)
            base_time = sunpy.time.parse_time(base_raw.meta['date-obs'], format='datetime')
            time_string=base_time.strftime("%Y%m%d-%H-%M-%S")
            basemap_path = os.path.join(basemap_folder,'uncropped_basemap-{string}.fits'.format(string=time_string))
            
            #if any([changes['wavelength'], changes['cadence'], changes['time_range']]) or any([changes['lat_val'], changes['lon_val'], changes['lat_dir'], changes['lon_dir']]):
            if not os.path.exists(basemap_path) or overwrite_files_calibration:
                if base_raw.meta['exptime']<1.5:
                    st.warning('Warning! Basemap exposure time is too low', icon="⚠️")
                basemap,base_time = mp.calibrate_map(base_raw)
                time_string=base_time.strftime("%Y%m%d-%H-%M-%S")
                basemap.save(basemap_path, overwrite=True)
                
            else:
                st.warning('Full Basemap already exists, checking for cropped version -')
                basemap =sunpy.map.Map(basemap_path)

            
            ##### Cropping Base Map
            basemap_cropped_path = os.path.join(basemap_folder,'cropped_basemap-{string}.fits'.format(string=time_string))
            if lon_dir=='East':
                lon_val = -lon_val
            if lat_dir=='South':
                lat_val = -lat_val
            centre_flare  = SkyCoord(lon_val*u.deg, lat_val*u.deg,   ## Lon/Lat
            frame="heliographic_stonyhurst",
            obstime=basemap.meta['date-obs'])
            cenX = centre_flare.transform_to(frames.Helioprojective(observer=basemap.coordinate_frame.observer)).Tx.to_value()
            cenY = centre_flare.transform_to(frames.Helioprojective(observer=basemap.coordinate_frame.observer)).Ty.to_value()
            
            
            #if any([changes['wavelength'], changes['cadence'], changes['time_range']]) or any([changes['lat_val'], changes['lon_val'], changes['lat_dir'], changes['lon_dir']]): 
            if not os.path.exists(basemap_cropped_path) or overwrite_files_calibration:  
                base_map_cropped = mp.crop_map(basemap,cenX,cenY)
                base_time = sunpy.time.parse_time(basemap.meta['date-obs'], format='datetime')
                time_string=base_time.strftime("%Y%m%d-%H-%M-%S")
                base_map_cropped.save(basemap_cropped_path ,overwrite=True)
                
                
                
            else:
                st.warning('Cropped Basemap already exists')
                base_map_cropped = sunpy.map.Map(basemap_cropped_path)
                
                


            st.success(f"Calibrated and Saved basemap files to: {basemap_folder}")

        progress_text = "Calibrating Data Files..."
        progress_bar = st.progress(0, text=progress_text)
        range_det = timedelta(minutes=time_range)
        cad_det = timedelta(seconds=cadence)

        tot_files = int(range_det/cad_det)

        tot_files_2 = len(listdir) - 1  # Exclude basemap image

        if tot_files > tot_files_2:
            total_files = tot_files_2
        else:
            total_files = tot_files



        
        for z in range(0,total_files):
    
        #for z in range(100, 101):
            try:
            # Update progress
                progress_percent = round(((z+1) / total_files) * 100)
                progress_bar.progress(progress_percent, 
                                text=f"{progress_text} {z+1}/{total_files} files")
                
            
        ## Your existing processing
                map_file = os.path.join(save_path_fits, listdir[z+1])
                if not len(os.listdir(diff_rot_folder))==total_files or overwrite_files_calibration:
                #if any([changes['wavelength'], changes['cadence'], changes['time_range']]):
                    map_el= sunpy.map.Map(map_file)
                    if map_el.meta['exptime']<1.5:
                        st.warning(f"Warning! {listdir[z+1]} exposure time is too low", icon="⚠️")
                    cal_map, t = mp.calibrate_map(map_el)
                    
                    time_diff = -(t - base_time).to(u.min)
                    
                    out_warp = mp.diff_rot(cal_map, time_diff, basemap)
                    
                    out_warp.save(os.path.join(diff_rot_folder, listdir[z+1]), overwrite=True)
        
                    
                    
                
            except Exception as e:
                st.warning(f"Error processing {listdir[z+1]}: {str(e)}")
                
            continue

        ## Completion
        progress_bar.progress(100, text=f"Completed {total_files}/{total_files} files")
        st.success(f"Calibration complete! Saved files to: {diff_rot_folder}")
        should_calibrate = False
    

        
    if should_LBR:
        st.header('Make Logarithmic Base Ratio (LBR images)')
        st.markdown('Here we create LBR images for our dimming detection using Base Difference images. Note that the saved '
                'LBR images will be cropped in an area of 1000X1000 arcsecs around the flare source')
    ## Creating Base Diff and LBR
        listdir = sorted(os.listdir(diff_rot_folder))
        progress_text = "Processing Log Difference Images..."
        progress_bar = st.progress(0, text=progress_text)
        #total_files = len(listdir)

        tot_files_2 = len(listdir)

        if tot_files > tot_files_2:
            total_files = tot_files_2
        else:
            total_files = tot_files
        
        for i in range(total_files):
            try:
                # Update progress (0-100%)
                progress_percent = int((i + 1) / total_files * 100)
                progress_bar.progress(
                    progress_percent,
                    text=f"{progress_text} {i+1}/{total_files} files"
                )
                
                # Process current file
                data_path = os.path.join(diff_rot_folder, listdir[i])
                lbr_path = os.path.join(log_diff_folder, listdir[i])
                if not len(os.listdir(log_diff_folder))==total_files or overwrite_files_lbr:
                    
                #if any([changes['wavelength'], changes['cadence'], changes['time_range']]) or any([changes['lat_val'], changes['lon_val'], changes['lat_dir'], changes['lon_dir']]):
                    data_map = sunpy.map.Map(data_path)
                    data_map_cropped = mp.crop_map(data_map,cenX,cenY,crop_size_x=1000,crop_size_y=1000)
                    
                    
                    
                
            # Calculate log difference
                    with np.errstate(divide='ignore', invalid='ignore'):  # Handle log(0)
                        lbr = np.log10(data_map_cropped.data) - np.log10(base_map_cropped.data)
                    #lbr = np.nan_to_num(lbr, nan=0.0, posinf=0.0, neginf=0.0)  # Clean invalid values
                    
                    logbr = sunpy.map.Map(lbr, data_map_cropped.meta)
                    logbr.save(os.path.join(log_diff_folder, listdir[i]), overwrite=True)
                
                
            except Exception as e:
                st.warning(f"Error processing {listdir[i]}: {str(e)}")
                continue

        # Final completion
        progress_bar.progress(100, text=f"Completed {total_files}/{total_files} files")
        st.success(f"Log difference processing complete! Saved files to: {log_diff_folder}")
        


        st.session_state.processed_data.update({
            'base_map_cropped': base_map_cropped,
            'listdir': sorted(os.listdir(log_diff_folder)),
            'log_diff_folder': log_diff_folder,
            'cenX': cenX,
            'cenY': cenY,
            'base_time': base_time,
            'diff_rot_folder':diff_rot_folder
        })
        st.session_state.data_processed = True
        should_LBR = False

    ##  LBR code...
        
    if should_detect:
        st.header('Running Dimming Detection')
        if changes['lbr_thresh'] or st.session_state.save_plots_checkbox:
        #st.session_state.prev_lbr_thresh = lbr_thresh
            data_old = st.session_state.processed_data
            base_map_cropped = data_old['base_map_cropped']
            listdir = data_old['listdir']
            log_diff_folder = data_old['log_diff_folder']
            cenX = data_old['cenX']
            cenY = data_old['cenY']
            base_time = data_old['base_time']
            diff_rot_folder= data_old['diff_rot_folder']
        st.info(f"""
        **Parameters:**  
        - **Base Map time:** `{base_time}`  
        - **Wavelength:** `{wavelength}`  
        - **Range of Detection:** `{time_range} minutes`  
        - **Cadence:** `{cadence}`  
        - **Flare source (Lat):** `{lat_val}° {lat_dir}`  
        - **Flare source (Lon):** `{lon_val}° {lon_dir}`  
        - **LBR Threshold:** `{lbr_thresh}`  
    """)
    
        save_path_detection = os.path.join(current_dir, 'Events', safe_event, 'Detection_figures',str(wavelength),str(cadence))
        os.makedirs(save_path_detection, exist_ok=True)
        save_path_timing = os.path.join(current_dir, 'Events', safe_event, 'Timing_Map',str(wavelength),str(cadence))
        os.makedirs(save_path_timing, exist_ok=True)
        
    
        #current_thresh = st.session_state.lbr_thresh
        data0_base = base_map_cropped
        cu_mask = data0_base.data*0.0
        time_pix = data0_base.data*np.nan
        
        element = np.array([[0,1,0], [1,1,1], [0,1,0]]) # cross element

        
        with st.spinner(f"Detecting Dimming...",show_time=True):
            progress_text = "Dimming Detection In progress..."
            progress_bar = st.progress(0, text=progress_text)
            total_files = len(listdir)
        
        
        
            #for i in range(len(listdir)):
            for i in range(0,total_files):
                progress_percent = int((i + 1) / total_files * 100)
            
                progress_bar.progress(
                    progress_percent,
                    text=f"{progress_text} {i+1}/{total_files} files"
                    )
            
            
                data = os.path.join(log_diff_folder, listdir[i])
                smap = sunpy.map.Map(data)
                
                #defining dimming region
                seeds_list=mp.find_seeds_LBR(smap.data, thresh=lbr_thresh,perc= 0.3)   # default values: threshold=-0.19, percentage=0.3
                seeds = np.array(seeds_list)
                seeds = ndimage.median_filter(seeds, size=10)
                segmented_image= mp.segment(seeds,smap.data, threshold=lbr_thresh)
                

                
                I_closed = binary_closing(segmented_image, element)
                segmented_image  = ndimage.binary_fill_holes(I_closed)
                seg_data = np.where(segmented_image!=0,smap.data,np.nan)
                segmented_map=sunpy.map.Map(seg_data, smap.meta)
                cu_mask[segmented_image]+=1
                arr = cu_mask ==1
                time_pix[arr] = mp.delta_from_base(smap, base_time = base_map_cropped.meta['date-obs'])
                time_pix_map = sunpy.map.Map(time_pix, smap.meta)
                



                data = os.path.join(diff_rot_folder, listdir[i])
                data0 = sunpy.map.Map(data)
                orig_2=data0
                orig=mp.crop_map(data0,cenX,cenY,crop_size_x=1000,crop_size_y=1000)


                # plotting 4 panels
                fig, axes = plt.subplots(1, 4, figsize=(15, 4), num=1, clear=True)
            # Define subplot data and settings
                subplot_data = [
                ((orig.data),'sdoaia211','Calibrated', 0, 1000, orig_2.plot_settings['norm']),
                ((smap.data),'gray', 'LBR', lbr_thresh, -lbr_thresh,smap.plot_settings['norm']),
                ((smap.data),'gray', 'LBR+Inst', lbr_thresh, -lbr_thresh,smap.plot_settings['norm']),
                ((time_pix_map.data),'jet', 'LBR+CU', 0, 50,time_pix_map.plot_settings['norm'])
                ]
                fig.suptitle(smap.meta['date-obs'])
            # Loop over subplots and set data and settings
                for ax, (data,cmap, title, vmin, vmax,norm) in zip(axes, subplot_data):
                    if ax==axes[0]:
                        norm.vmin=vmin
                        norm.vmax=vmax
                        ax.imshow(data, cmap=cmap,norm=norm,origin="lower")
                    else:
                        ax.imshow(smap.data, cmap='gray',vmin=lbr_thresh,vmax=-lbr_thresh,origin="lower")
                        ax.imshow(data, cmap=cmap,vmin=vmin,vmax=vmax,origin="lower")


                    ax.set_title(title)
                    ax.axis('off')

                ax3 = axes[2]
                ax3.imshow((segmented_map.data),cmap='copper',vmin=-10000,origin="lower")
                if save_plots_checkbox:
                    
                    fig.savefig(os.path.join(save_path_detection, listdir[i][:-5]+'.png'),facecolor='w',bbox_inches='tight')
                if i==total_files-1:
                    st.write('Final step map')
                    st.pyplot(fig)
                    if save_plots_checkbox_all:
                        fig.savefig(os.path.join(save_path_detection, listdir[i][:-5]+'.png'),facecolor='w',bbox_inches='tight')

            
            if save_plots_checkbox:
                st.success(f'All detection plots saved in {save_path_detection}') 
            time_pix_map.save(os.path.join(save_path_timing, listdir[i]),overwrite=True)
            st.session_state.thresh_changed = False  # Reset change flag
            st.success("Dimming Detected!")
            
        with st.spinner(f"Generating Detection Video (Threshold: {lbr_thresh})...",show_time=True):
            detection_video_path = os.path.join(current_dir, 'Events', safe_event, 'Detection_video',str(wavelength),str(cadence))
            os.makedirs(detection_video_path, exist_ok=True)
            img_clips = []
            path_list=[]

            #accessing path of each image
            for image in os.listdir(save_path_detection):
                if image.endswith(".png"):
                    path_list.append(os.path.join(save_path_detection, image))

            #creating slide for each image
            for img_path in path_list:
                slide = ImageClip(img_path,duration=0.1)
                img_clips.append(slide)

            #concatenating slides
            video_slides = concatenate_videoclips(img_clips, method='compose')
            #exporting final video
            video_slides.write_videofile(os.path.join(detection_video_path, "detection_video.mp4"), fps=16)
            st.success('Video Generated')

        # mp.close_file(base_raw)
        # mp.close_file(basemap)
    

        with st.spinner(f"Calculating Dimming area growth and impulsive phase data...", show_time=True):
            path_2 = os.path.join(current_dir, 'Events', safe_event, 'variables',str(wavelength),str(cadence))
            os.makedirs(path_2, exist_ok=True)
        
            file_list_mask = sorted(os.path.join(save_path_timing, f) for f in os.listdir(save_path_timing))

    # Use generator for memory efficiency when creating maps
            final_image_mask = [sunpy.map.Map(f) for f in file_list_mask]

        

            # Calculate differences on the last image
            last_data = final_image_mask[-1].data.astype('float32')
            differences = np.round(np.diff(last_data))

            # Display unique values (assuming Streamlit)
            

            # Initialize other variables (if actually needed)
            final_time_str, area, num, count,t_pix_old = [], [0], [], 0,0
            
            
            # Find the smallest non-zero difference
            try:
                stp = np.nanmin(differences[differences > 0])
            
                #for t_pix in np.arange(np.nanmin(final_image_mask[-1].data), np.nanmin(final_image_mask[-1].data)+stp+stp+stp,stp):
                for t_pix in np.arange(np.nanmin(final_image_mask[-1].data), np.nanmax(final_image_mask[-1].data)+stp,stp):
                    
                    if t_pix in np.unique(np.round(final_image_mask[-1].data)):
                        
                                
                        
                        image_mask = np.where(np.round(final_image_mask[-1].data) <= t_pix, np.round(final_image_mask[-1].data), np.nan)
                        image_mask_map = sunpy.map.Map(image_mask, final_image_mask[-1].meta)
                        
                        result = np.where((np.round(image_mask_map.data) <= t_pix) & (np.round(image_mask_map.data) > t_pix_old))
                        area_int = mp.calculate_region_area_2(result, image_mask_map) + area[-1]
                        area.append(area_int)
                        t_pix_old = t_pix
                        
                        final_time = base_time + timedelta(minutes=t_pix)
                        final_time_str.append(final_time.strftime('%m/%d/%Y %H:%M:%S.%f'))
                        num.append(count)
                        count = count+1
                    else:
                        continue
                area.pop(0)
                smooth_area = mp.exp_smooth(np.array(area), alpha=0.25)


    
    # Pre-compute derivatives using numpy (faster than pandas diff())
                area_der = np.diff(smooth_area, prepend=np.nan)
                area_der_un = np.diff(area, prepend=np.nan)

                # Convert timestamps in bulk before DataFrame creation
                times = pd.to_datetime(final_time_str)

                # Single-step DataFrame creation with all columns
                df = pd.DataFrame({
                    'Num': num,
                    'area': smooth_area,
                    'area_un': area,
                    'time': final_time_str,
                    'times': times,
                    'area_der': area_der,
                    'area_der_un': area_der_un
                })

                df_2 = pd.DataFrame({
                    'Data/Time of Event': times,
                    'Area Smoothed': smooth_area,
                    'Derivative of Area - Smoothed': area_der,
                    'Area Unsmoothed': area,
                    'Derivative of Area - Unsmoothed': area_der_un
                })



                max_impulsive_idx = np.nanargmax(area_der)
                max_impulsive_time = times[max_impulsive_idx]
                
                # Pre-compute threshold
                q = df[(df['area_der'] <= 0.15 * df['area_der'][max_impulsive_idx]) & (df['Num'] >= max_impulsive_idx)][:1]

                if q.empty:
                    q = df.tail(1)

                end_impulsive_idx = q['area_der'].idxmax()
                end_impulsive_time = q['times'][end_impulsive_idx]
            

                rr = q['times'].to_numpy()[0]
                end_time_dt = pd.to_datetime(rr)

                

                with open(os.path.join(path_2, 'end_impulsive_phase.pickle'), 'wb') as handle:
                    pickle.dump(end_impulsive_time, handle)
                with open(os.path.join(path_2,'end_impulsive_idx.pickle'), 'wb') as handle:
                    pickle.dump(end_impulsive_idx, handle)
                with open(os.path.join(path_2, 'max_impulsive_phase.pickle'), 'wb') as handle:
                    pickle.dump(max_impulsive_time, handle)
                with open(os.path.join(path_2, 'max_impulsive_idx.pickle'), 'wb') as handle:
                    pickle.dump(max_impulsive_idx, handle)
                with open(os.path.join(path_2, 'begin_impulsive_phase.pickle'), 'wb') as handle:
                    pickle.dump(df['times'][0], handle)
                with open(os.path.join(path_2, 'begin_time.pickle'), 'wb') as handle:
                    pickle.dump(pd.to_datetime(base_map_cropped.meta['date-obs']), handle)
                with open(os.path.join(path_2, 'end_time.pickle'), 'wb') as handle:
                    pickle.dump(df['times'][df.index[-1]], handle)
                area_path = os.path.join(current_dir, 'Events', safe_event,'Area.txt')

                st.write(f'Max Impulsive time: {str(max_impulsive_time)}')
                st.write(f'End Impulsive time: {str(end_impulsive_time)}')
                
                buffer = StringIO()
                df_2.to_string(buffer)
                text_data = buffer.getvalue()

                with open(area_path, 'w') as f:
                    f.write(text_data)
                st.success(f' Area info saved in {area_path}')
                st.info(f' Generating plot...')
                
        
                df1_plot = df.copy()
                df1_plot['area'] = df1_plot['area']/10**10
                df1_plot['area_der'] = df1_plot['area_der']/10**10

                # Create template
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
                
                # Add traces
                fig.add_trace(
                    go.Scatter(
                        x=df1_plot['times'], 
                        y=df1_plot['area'],
                        mode='lines', 
                        line=dict(color='blue'),
                        name='Area'
                    ), 
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df1_plot['times'], 
                        y=df1_plot['area_der'],
                        mode='lines', 
                        line=dict(color='red'),
                        name='Area Derivative'
                    ), 
                    row=2, col=1
                )

                # Update layout
                fig.update_layout(
                    title='',
                    height=600,
                    template='plotly_white',
                    showlegend=False,
                    margin=dict(l=20, r=20, t=20, b=20)
                )

                # Update axes
                fig.update_yaxes(
                    title_text='Dimming Area [10<sup>10</sup> km<sup>2</sup>]', 
                    row=1, col=1,
                    mirror=True,
                    ticks='inside',
                    zeroline=False,
                    showline=True,
                    linewidth=2,
                    title_font=dict(size=16, family='Arial'),
                    tickwidth=2,
                    tickfont=dict(size=14),
                    linecolor='blue',
                    gridcolor='grey'
                )

                fig.update_yaxes(
                    title_text="Derivative [10<sup>10</sup> km<sup>2</sup>]", 
                    row=2, col=1,
                    mirror=True,
                    ticks='inside',
                    zeroline=False,
                    showline=True,
                    linewidth=2,
                    title_font=dict(size=16, family='Arial'),
                    tickwidth=2,
                    tickfont=dict(size=14),
                    linecolor='red',
                    gridcolor='grey'
                )

                fig.update_xaxes(
                    row=1, col=1,
                    nticks=8,
                    title_text="",
                    mirror=True,
                    ticks='inside',
                    tickformat="%H:%M:%S",
                    showline=True,
                    linewidth=2,
                    title_font=dict(size=14, family='Arial'),
                    tickwidth=2,
                    tickfont=dict(size=12),
                    linecolor='grey',
                    gridcolor='grey'
                )

                fig.update_xaxes(
                    row=2, col=1,
                    nticks=8,
                    mirror=True,
                    ticks='inside',
                    tickformat="%H:%M:%S",
                    showline=True,
                    linewidth=2,
                    title_font=dict(size=14, family='Arial'),
                    tickwidth=1,
                    tickfont=dict(size=12),
                    linecolor='grey',
                    gridcolor='grey'
                )

                # Add reference lines and annotations
                fig.add_hline(
                    y=0.15*df1_plot['area_der'].max(),
                    line_dash="dash", 
                    line_color='magenta',
                    row=2, col=1
                )
                
                fig.add_vline(
                    x=df1_plot['times'][q['Num'].to_numpy()[0]],
                    line_dash="dash", 
                    line_color='magenta',
                    row=2, col=1
                )

                fig.add_annotation(
                    x=df1_plot['times'][q['Num'].to_numpy()[0]] - timedelta(hours=0, minutes=10), 
                    y=0.15*df1_plot['area_der'].max(),
                    row=2, col=1,
                    text="15% of max",
                    showarrow=False,
                    font=dict(family='Arial', size=14, color='magenta')
                )

                fig.add_annotation(
                    x=df1_plot['times'][q['Num'].to_numpy()[0]], 
                    y=0.15*df1_plot['area_der'].max(),
                    row=2, col=1,
                    text=end_time_dt.strftime('%H:%M:%S'),
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=2,
                    font=dict(family='Arial', size=14, color='magenta'))

                #st.set_page_config(layout="wide")    
                st.plotly_chart(fig, use_container_width=True)
                if save_plots_checkbox_all:
                    fig.savefig(os.path.join(save_path_plots, 'area_derivative'+'.png'),facecolor='w',bbox_inches='tight')

                
                st.success(f'Plot generated')
                


            except Exception as e:
                st.warning(f"Error processing")
            should_detect = False










    
    
    


    

   


