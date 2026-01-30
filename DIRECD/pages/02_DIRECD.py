import streamlit as st
from functions import map_calibration as mp
from functions import direcd_functions as direcd
import os
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.path
import numpy as np
import sunpy.map
from io import BytesIO
import astropy.units as u
from sunpy.coordinates import Helioprojective, HeliographicStonyhurst,NorthOffsetFrame, frames
from astropy.coordinates import SkyCoord
import pickle
import pandas as pd
from datetime import datetime, timedelta
from scipy import ndimage, misc
from skimage.morphology import rectangle
from itertools import cycle, islice, dropwhile
import mplcursors
import streamlit.components.v1 as components
import skimage.measure
import math
from io import StringIO
import plotly.graph_objects as go
import matplotlib.gridspec as gridspec
from astropy.coordinates.representation import CartesianRepresentation
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage,
                                  TextArea)
from plotly.subplots import make_subplots
import matplotlib.image as image
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
#import kaleido

current_dir = Path(__file__).parent.absolute().parent
logo_folder_short = os.path.join(current_dir, 'logo', 'direcd_short.png')
st.set_page_config(page_title="Application of DIRECD", page_icon=logo_folder_short)
st.header("DIRECD: Dimming Inferred Estimation of CME Direction")

st.markdown('---')


if 'submitted_2' not in st.session_state:
    st.session_state.submitted_2 = False

if 'wavelength_val' not in st.session_state:
        st.session_state.wavelength_val = 211
if 'cadence_val' not in st.session_state:
    st.session_state.cadence_val = 60
if 'prev_lon' not in st.session_state:
    st.session_state.prev_lon = 0
if 'prev_lon_dir' not in st.session_state:
    st.session_state.prev_lon_dir = 'West'
if 'prev_lat' not in st.session_state:
    st.session_state.prev_lat = 0
if 'prev_lat_dir' not in st.session_state:
    st.session_state.prev_lat_dir = 'North'
if 'first_submit' not in st.session_state:
    st.session_state.first_submit = True
if 'time_to_analyze_map' not in st.session_state:
    st.session_state.time_to_analyze_map = 'End of Impulsive Phase'
if 'manual_edge_detection_checkbox' not in st.session_state:
    st.session_state.manual_edge_detection_checkbox = False
if 'edge_coords' not in st.session_state:
    st.session_state.edge_coords = {'edge1_x': None, 'edge1_y': None, 
                                  'edge2_x': None, 'edge2_y': None}
if 'ensemble_cones' not in st.session_state:
    st.session_state.ensemble_cones = False

if 'submit_edge' not in st.session_state:
    st.session_state.submit_edge = False
if 'folder_path' not in st.session_state:
        st.session_state.folder_path = False


default_date = "today"
default_time = '20:44'
default_flare_lat = 0
default_index_lat = 0
default_flare_lon = 0
default_index_lon = 0

if st.session_state.folder_path is False:
    def select_folder():
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(master=root)
        root.destroy()
        return folder_path

    selected_folder_path = st.session_state.get("folder_path", None)
    folder_select_button = st.button("Please select a folder for where detection results are saved")
    if folder_select_button:
        selected_folder_path = select_folder()
    if selected_folder_path:
        st.write('Selected folder path:', selected_folder_path)
    st.session_state.folder_path = selected_folder_path


    
    
if st.session_state.folder_path is not False:
    selected_folder_path = st.session_state.folder_path
    default_date = "today"
    default_time = 'now'
    default_flare_lat = 0
    default_index_lat = 0
    default_flare_lon = 0
    default_index_lon = 0

    try:
        data_event = pd.read_csv(os.path.join(st.session_state.folder_path, 'list_events.csv'))
        string_options = (data_event['Event'] + 'T' + data_event['Time']).str.replace('-', '/').tolist()
        string_option_1 = 'New Event'
        string_options.insert(0, string_option_1)
    except:
        string_options='New Event'

    list_events =st.sidebar.selectbox('Event', options = (string_options), index=0,
        placeholder="list_events")
    st.session_state.list_events = list_events

   

    if st.session_state.list_events != 'New Event':
        default_date = datetime.strptime(st.session_state.list_events, '%d/%m/%YT%H:%M').date()
        default_time = datetime.strptime(st.session_state.list_events, '%d/%m/%YT%H:%M').time()
        matching_rows = data_event[
        (data_event['Event'] == default_date.strftime('%d-%m-%Y')) & 
        (data_event['Time'] == default_time.strftime('%H:%M'))
        ]
        default_flare_lat = abs(matching_rows['Lat'].squeeze())
        default_flare_lon = abs(matching_rows['Lon'].squeeze())
        if matching_rows['Lat'].squeeze()<0:
            default_index_lat = 1
        else:
            default_index_lat = 0
    
        if matching_rows['Lon'].squeeze()<0:
            default_index_lon = 1
        else:
            default_index_lon = 0






    form = st.sidebar.form("Event_selection")
    date_col, time_col = form.columns([5, 4])
    date_str_min = "01/01/2009"
    parsed_date = datetime.strptime(date_str_min, "%d/%m/%Y").date()

    help_txt = '''
    # Start date of the event

    '''
    with date_col:
        date_event = st.date_input("Date",value=default_date,min_value=parsed_date,key='input_date',help=help_txt)  

    help_txt = '''
    # Start time of the event  
    (Flare Start time/Start time of the period to be analyzed)

    '''
    with time_col:
        time_event = st.time_input("Time",value=default_time,step=60,key='input_time',help=help_txt) 


    wavelength_col, sample_col = form.columns([2, 2])

    help_txt = '''
    # SDO/AIA Wavelength

    '''
    with wavelength_col:
        wavelength = st.selectbox(
        "Wavelength (A)",
        (193, 211),
        index=1,
        placeholder="Wavelength",help=help_txt
        
    )

    with sample_col:
        cadence = st.selectbox(
        "Cadence (seconds)",
        (60,120,180,240),
        index=0,
        placeholder="Cadence (seconds)",
        
    )

    lat_value, lat_direction = form.columns([2, 2])

    help_txt = '''
    # Latitude of Flare Source Coordinates (in HEEQ)

    '''

    with lat_value:
        lat_val = st.slider("Flare Source Lat", 0, 60, default_flare_lat,help=help_txt)

    with lat_direction:
        lat_dir = st.selectbox(" ",("North", "South"),index = default_index_lat)

    help_txt = '''
    # Latitude of Flare Source Coordinates (in HEEQ)

    '''
        
    lon_value, lon_direction = form.columns([2, 2])

    with lon_value:
        lon_val = st.slider("Flare Source Lon", 0, 60, default_flare_lon,help=help_txt)
    with lon_direction:
        lon_dir = st.selectbox(" ",("West", "East"),index=default_index_lon,
    )
    numeric_options = list(np.arange(31, 151,1))
    string_options = 'End of Impulsive Phase'
    numeric_options.insert(0, string_options)

    help_txt = '''
    # Minutes Since Event Start  
    (Note: Count begins from a base-map 30 minutes prior to event start, so "31" = 1 minute after start).

    '''

    time_analyze = form.selectbox(
        "Time to Analyze Map - Minutes after Base map",
        options= (numeric_options),
        index=0,
        placeholder="Time to Analyze Map",help=help_txt
        
    )

    help_txt = '''
    Check this box to manually define the edges of dimming for cone generation.  
    If unsure, leave this box unchecked and the system will find them automatically.

    '''
    manual_edge_detection_checkbox = form.checkbox('Manual Edge Detection',label_visibility="visible", help=help_txt)

    help_txt = '''
    Auto-save plots. When disabled, use right-click 'Save As' to save manually
    '''

    save_all_plots_checkbox = form.checkbox('Save all plots',label_visibility="visible", help=help_txt)

    help_txt = '''
    #To plot ensemble of 20 cones

    Check this box to plot cones with different height, width and inclination angles.
    '''

    plot_ensemble_cones = form.checkbox('Plot ensemble of cones',label_visibility="visible", help=help_txt)

    submit = form.form_submit_button("Submit")


    current_dir = Path(__file__).parent.absolute().parent

    processing_map = False
    edge_selection=False
    generate_cones=False






    def detect_parameter_changes(wavelength, cadence,lon_val,lon_dir,lat_val,lat_dir,time_analyze,manual_edge_detection_checkbox):
        # Initialize change tracking
        changes = {
            'any_changed': False,
            'wavelength': False,
            'cadence': False,
            'lon_val': False,
            'lon_dir': False,
            'lat_val': False,
            'lat_dir': False,
            'time_to_analyze_map': False,
            'manual_edge_detection_checkbox': False,
            'changed_params': []
        }
        
        # Check each parameter
        if st.session_state.wavelength_val != wavelength:
            st.session_state.wavelength_val = wavelength
            changes['wavelength'] = True
            changes['changed_params'].append('wavelength')
        
        if st.session_state.cadence_val != cadence:
            st.session_state.cadence_val = cadence
            changes['cadence'] = True
            changes['changed_params'].append('cadence')
        
        
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
        if st.session_state.time_to_analyze_map != time_analyze:
            st.session_state.time_to_analyze_map = time_analyze
            changes['time_to_analyze_map'] = True
            changes['changed_params'].append('Time to Analyze Map')
        if st.session_state.manual_edge_detection_checkbox != manual_edge_detection_checkbox:
            st.session_state.manual_edge_detection_checkbox = manual_edge_detection_checkbox
            changes['manual_edge_detection_checkbox'] = True
            changes['changed_params'].append('manual_edge_detection_checkbox')
        
        
        
        # Set overall change flag
        changes['any_changed'] = len(changes['changed_params']) > 0
        
        return changes

    changes = detect_parameter_changes(wavelength, cadence,lon_val,lon_dir,lat_val,lat_dir,time_analyze,manual_edge_detection_checkbox)

    if submit:
        st.session_state.submitted_2 = True
    else:
        st.write('Please click submit to run this page')
    if st.session_state.submitted_2:

        if plot_ensemble_cones :
            st.session_state.ensemble_cones = True
        else:
            st.session_state.ensemble_cones = False
        

        event_dt = mp.create_event_datetime(date_event, time_event)
        safe_event = event_dt.strftime("%Y-%m-%dT%H:%M:%S").replace(":", "-")
        if save_all_plots_checkbox:
            save_path_plots = os.path.join(selected_folder_path, 'Events', safe_event, 'plots',str(wavelength),str(cadence))
            os.makedirs(save_path_plots, exist_ok=True) 
        if changes['any_changed']:
            st.write(f"Parameters changed: {', '.join(changes['changed_params'])}")
            if any([changes['manual_edge_detection_checkbox']]):
                edge_selection=True
                processing_map = True
                
            elif any([changes['wavelength'], changes['cadence'],changes['lat_val'], changes['lat_dir'],changes['lon_val'],changes['lon_dir'],changes['time_to_analyze_map']]):
                processing_map = True
                edge_selection=True
                
        else:
            edge_selection=True
            processing_map = True
            










    if processing_map:
        st.header('Processing Timing Map ')
        st.markdown('Here we process end of imppulsive phase timing map and plot it'
                    )
            
        
        with st.spinner(f"Processing End of Impulsive Phase Map",show_time=True):
            
            main_event_folder = os.path.join(selected_folder_path, 'Events', safe_event)
            save_path_timing = os.path.join(selected_folder_path, 'Events', safe_event, 'Timing_Map',str(wavelength),str(cadence))

            list_dir_map = sorted(os.listdir(save_path_timing))
            files = [os.path.join(save_path_timing, f) for f in list_dir_map] # add path to each file
            list_dir_map = sorted(files,key=os.path.getmtime)
            st.write(save_path_timing)
            file_event=os.path.join(save_path_timing,list_dir_map[-1])
            final_timing_map =  sunpy.map.Map(file_event)
            smap = final_timing_map


            

            if lon_dir=='East':
                lon_val = -lon_val
            if lat_dir=='South':
                lat_val = -lat_val

            try:
                Rsun = smap.meta['rsun_ref']
            except:
                Rsun=696000000

            ccc  = SkyCoord(lon_val*u.deg, lat_val*u.deg,Rsun*u.m,   ## Lon/Lat
            frame="heliographic_stonyhurst",
            obstime=smap.meta['date-obs'])


            center= SkyCoord(0*u.deg, 0*u.deg, 0*u.km,
                    frame="heliographic_stonyhurst",
                    obstime=smap.meta['date-obs'], observer='earth')
        
            north=ccc
            new_frame = NorthOffsetFrame(north=north)
            x,y = np.meshgrid(*[np.arange(v.value) for v in smap.dimensions]) * u.pix
            z = smap.pixel_to_world(x, y)
            all_map = SkyCoord(z,frame=Helioprojective)
            h = all_map.transform_to(new_frame)
            

            st.info(f'Flare Source coordinates are -  Lon: {round(ccc.lon.to_value())} , Lat: {round(ccc.lat.to_value())}')


            variables_folder = os.path.join(main_event_folder, 'variables',str(wavelength),str(cadence))
            os.makedirs(variables_folder, exist_ok=True)
            path_2 = variables_folder

            basemap_folder = os.path.join(selected_folder_path, 'Events', safe_event, 'basemap',str(wavelength),str(cadence))
            file_list_mask = sorted(os.path.join(basemap_folder, f) for f in os.listdir(basemap_folder))
            list_dir_map = sorted(file_list_mask,key=os.path.getmtime)
        # Use generator for memory efficiency when creating maps
            base_image_mask = [sunpy.map.Map(f) for f in list_dir_map]
            base_time = sunpy.time.parse_time(base_image_mask[-1].meta['date-obs'], format='datetime')
            time_string=base_time.strftime("%Y%m%d-%H-%M-%S")
            basemap_cropped_path = os.path.join(basemap_folder,'cropped_basemap-{string}.fits'.format(string=time_string))
            

            with st.spinner(f"Calculating Dimming area growth and impulsive phase data...", show_time=True):
                

            
                file_list_mask = sorted(os.path.join(save_path_timing, f) for f in os.listdir(save_path_timing))
                list_dir_map = sorted(file_list_mask,key=os.path.getmtime)

        # Use generator for memory efficiency when creating maps
                final_image_mask = [sunpy.map.Map(f) for f in list_dir_map]

            

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
                        
                        if round(t_pix) in np.unique(np.round(final_image_mask[-1].data)):
                            

                            image_mask = np.where(np.round(final_image_mask[-1].data) <= round(t_pix), np.round(final_image_mask[-1].data), np.nan)
                            image_mask_map = sunpy.map.Map(image_mask, final_image_mask[-1].meta)
                            
                            result = np.where((np.round(image_mask_map.data) <= round(t_pix)) & (np.round(image_mask_map.data) > round(t_pix_old)))
                            area_int = mp.calculate_region_area_2(result, image_mask_map) + area[-1]
                            area.append(area_int)
                            t_pix_old = t_pix
                            
                            final_time = base_time + timedelta(minutes=round(t_pix))
                            final_time_str.append(final_time.strftime('%m/%d/%Y %H:%M:%S.%f'))
                            num.append(count)
                            count = count+1
                        else:
                            continue
                    # for t_pix in np.arange(np.nanmin(final_image_mask[-1].data), np.nanmax(final_image_mask[-1].data)+stp,stp):
                        
                    #     if t_pix in np.unique(np.round(final_image_mask[-1].data)):
                            
                                    
                    #         st.write(t_pix)
                    #         image_mask = np.where(np.round(final_image_mask[-1].data) <= t_pix, np.round(final_image_mask[-1].data), np.nan)
                    #         image_mask_map = sunpy.map.Map(image_mask, final_image_mask[-1].meta)
                            
                    #         result = np.where((np.round(image_mask_map.data) <= t_pix) & (np.round(image_mask_map.data) > t_pix_old))
                    #         area_int = mp.calculate_region_area_2(result, image_mask_map) + area[-1]
                    #         area.append(area_int)
                    #         t_pix_old = t_pix
                            
                    #         final_time = base_time + timedelta(minutes=t_pix)
                    #         final_time_str.append(final_time.strftime('%m/%d/%Y %H:%M:%S.%f'))
                    #         num.append(count)
                    #         count = count+1
                    #     else:
                    #         continue
                    area.pop(0)
                   
                    alpha =[0.4, 0.25, 0.3]
                    for alpha_choose_2 in alpha:
                        smooth_area = mp.exp_smooth(np.array(area), alpha=alpha_choose_2)


    
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

                        time_hours = (end_impulsive_time - pd.to_datetime(base_image_mask[-1].meta['date-obs'])).total_seconds() / 3600

                        if time_hours>2.9:
                            st.write(time_hours)
                            continue
                        else:
                            break
                    
                    
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
                        pickle.dump(pd.to_datetime(base_image_mask[-1].meta['date-obs']), handle)
                    with open(os.path.join(path_2, 'end_time.pickle'), 'wb') as handle:
                        pickle.dump(df['times'][df.index[-1]], handle)
                    area_path = os.path.join(selected_folder_path, 'Events', safe_event,'Area.txt')

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
                        linecolor='grey'
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
                        linecolor='grey'
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
                        linecolor='grey'
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
                        linecolor='grey'
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
                    if save_all_plots_checkbox:
                        fig.write_image(file = os.path.join(save_path_plots, 'area_derivative'+'.png'),format='png')

                    
                    st.success(f'Plot generated')
                    


                except Exception as e:
                    st.warning(str(e))










            with open(os.path.join(variables_folder,'begin_impulsive_phase.pickle'), 'rb') as handle:
                begin_impulsive_phase = pickle.load(handle)
            impulse_begin_time = datetime(begin_impulsive_phase.year, begin_impulsive_phase.month, 
                                begin_impulsive_phase.day, begin_impulsive_phase.hour, 
                                begin_impulsive_phase.minute,begin_impulsive_phase.second)
            

            with open(os.path.join(variables_folder,  'begin_time.pickle'), 'rb') as handle:
                        begin_time_raw = pickle.load(handle)
            begin_time = datetime(begin_time_raw.year, begin_time_raw.month, begin_time_raw.day, 
                                begin_time_raw.hour, begin_time_raw.minute, begin_time_raw.second)
            st.info(f'Base Map Time: {begin_time}')
            with open(os.path.join(variables_folder, 'end_time.pickle'), 'rb') as handle:
                        end_time_raw = pickle.load(handle)
            end_time = datetime(end_time_raw.year, end_time_raw.month, end_time_raw.day, 
                                end_time_raw.hour, end_time_raw.minute,end_time_raw.second)
            if time_analyze == 'End of Impulsive Phase':
                with open(os.path.join(variables_folder, 'end_impulsive_phase.pickle'), 'rb') as handle:
                    end_impulsive_phase = pickle.load(handle)
                impulse_end_time = datetime(end_impulsive_phase.year, end_impulsive_phase.month, 
                                        end_impulsive_phase.day, end_impulsive_phase.hour, 
                                        end_impulsive_phase.minute,end_impulsive_phase.second)
            else:

                impulse_end_time = begin_time + timedelta(minutes=int(time_analyze))
            
            time_elapsed_end = impulse_end_time - begin_time
            qq1 = np.copy(smap.data)   
            qq1 = np.where(qq1 <= ((time_elapsed_end).total_seconds()) / 60, qq1, np.nan) 
            smap_end =sunpy.map.Map(qq1, smap.meta)

            try:
                st.write('Plotting Map')
                fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': smap_end})
                norm = Normalize(vmin=0, vmax=np.nanmax(smap.data))
                im= ax.imshow(smap_end.data, origin='lower', cmap='jet', norm=norm)

                ax.plot_coord(ccc,'o', color = 'red',markersize=10)
                plt.xlabel('X (Arcsec)')
                plt.ylabel('Y (Arcsec)')
                smap_end.draw_limb(axes=ax,color='red')
                cbar = plt.colorbar(im, ax=ax,orientation="horizontal")
                cbar.set_label('Minutes after start time')
                ax.plot_coord(ccc,'o', color = 'red',markersize=10)
                # Force uniform ticks (0, 20, 40, ..., 180)
                cbar.set_ticks(range(0, int(np.nanmax(smap_end.data))+1, 20))

                overlay = ax.get_coords_overlay(new_frame)
                overlay[0].set_ticks(np.arange(start=-15, stop=345, step=30) * u.deg, direction='in', alpha=0)
                overlay[1].set_ticks(spacing=10 * u.deg, direction='in', alpha=0)
                overlay[0].set_ticklabel(color='white', exclude_overlapping=True)
                overlay[1].set_ticklabel(color='white', exclude_overlapping=True)
                overlay[0].grid(ls='-', color='mediumblue', linewidth=0.5)
                overlay[1].grid(ls='--', color='lightgray', linewidth=0.5)
                overlay[0].set_axislabel(' ', minpad=1)
                overlay[1].set_axislabel(' ', minpad=1)
                overlay[0].set_ticks_visible('False')

                ax.tick_params(axis='x', which='both', top=False, bottom=True, direction='in')
                ax.tick_params(axis='y', which='both', right=False, left=True, direction='in')
                ax.set_title((begin_time + timedelta(seconds=time_elapsed_end.seconds)).strftime('%d-%b-%y %H:%M'), fontsize=14)
                ## Mark the centre coordinate
                #buf = BytesIO()
                st.pyplot(fig)
                plt.close(fig)
            

                if save_all_plots_checkbox:
                    fig.savefig(os.path.join(save_path_plots, 'end_of_implusive_phase_dimming'+'.png'),facecolor='w',bbox_inches='tight')
                    st.success(f'Plot saved in {save_path_plots} as end_of_implusive_phase_dimming.png')
            except Exception as e: 
                st.write(e)
                



        with st.spinner(f"Calculating Areas:",show_time=True):
            new_dimensions = [1024, 1024] * u.pixel
            #shape=[1024,1024]
            
            smap_end_full = direcd.full_disk_mask1(smap_end)
            old_dimensions = [smap_end_full.meta['naxis1'], smap_end_full.meta['naxis2']]*u.pixel


            smap_end_full = smap_end_full.resample(new_dimensions)
        
            smap_new = smap_end_full

            smap_mask_data = np.logical_not((np.isnan(smap_end_full.data)))
            smap_mask_2= sunpy.map.Map(smap_mask_data, smap_end_full.meta) 
            smap_cropped = smap_end

            rsun = smap_mask_2.meta['rsun_ref']*u.m

            img_fill_holes=ndimage.binary_fill_holes(smap_mask_2.data,structure=rectangle(2,3))
            smap_mask_2 = sunpy.map.Map(img_fill_holes,smap_mask_2.meta)
            smap_mask_data = direcd.eliminate_small_regions(smap_mask_2.data)
            smap_mask = sunpy.map.Map(smap_mask_data, smap_mask_2.meta)

            areamap_folder = os.path.join(main_event_folder, 'areamap',str(wavelength),str(cadence))
            os.makedirs(areamap_folder,exist_ok=True)

            area_map = mp.area_mask(smap_mask)   
            amap=direcd.full_disk_mask2(area_map)
            amap.save(os.path.join(areamap_folder, 'areamap.fits'),overwrite=True)
            sector1n,sector2n,sector3n,sector4n,sector5n,sector6n,sector7n,sector8n,sector9n,sector10n,sector11n,sector12n = direcd.choosing_sectors(h)
            sectors = [sector1n, sector2n, sector3n, sector4n, sector5n, sector6n,
                sector7n, sector8n, sector9n, sector10n, sector11n, sector12n]

            areas = [direcd.sectors_to_pix_and_data(sector, smap_mask, area_map=amap)[0] 
                    for sector in sectors]

            direcd.reset_pixel_tracking()
            area1, area2, area3, area4, area5, area6, \
            area7, area8, area9, area10, area11, area12 = areas

            frames_sector = []

            for i in range(1, 13):
                sector = pd.DataFrame({"sector": i,
                                    "sec_deg": (i - 1) * 30,
                                    "area": globals()[f"area{i}"]
                                    })
                frames_sector.append(sector)

            sectorz = pd.concat(frames_sector)
            sectorz['area_plot'] = sectorz.area/10**9
            total_pix = sectorz.groupby(['sector'],as_index=False).sum()
            max_sector = total_pix['sector'][total_pix['area'].idxmax()]

            st.info(f'Max Sector: {max_sector}')

            sectorz['area_plot'] = sectorz['area'] / 1e9

            st.write('Generating Area-Sector Plot')
            fig_2, (ax) = plt.subplots(1, 1, figsize=(5, 3))

            total_pix.plot(x='sector', y='area_plot', ax=ax, kind='scatter', color='seagreen', legend=False)
            total_pix.plot(x ='sector',y='area_plot',ax=ax,c='seagreen')
            #total_pix.plot(x='sector', y='intensity', ax=ax2, kind='scatter', color='seagreen', legend=False)
            #total_pix.plot(x ='sector',y='intensity',ax=ax2,c='seagreen')
            ax.set_xlabel("Sector", fontsize=16)
            ax.set_xticks(range(1, 13))
            ax.tick_params(axis='both', labelsize=14)

            ax.set_ylabel("Area ($10^9$ $km^2$)", fontsize=16)
            #ax2.set_ylabel("Intensity", fontsize=16)
            ax.legend('')
            #ax2.legend('')
            fig_2.tight_layout()

            #buf = BytesIO()
            # plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=300)
            #   # Important: close the figure to free memory
            
            # st.image(buf)
            # plt.close(fig_2)
            st.pyplot(fig_2)
            plt.close(fig_2)

            if save_all_plots_checkbox:
                fig_2.savefig(os.path.join(save_path_plots, 'end_of_implusive_phase_areas'+'.png'),facecolor='w',bbox_inches='tight',dpi=300)
                st.success(f'Plot saved in {save_path_plots} as end_of_implusive_phase_areas.png')
        


            area_dict = {1: area1, 2: area2, 3: area3, 4: area4, 5: area5, 6: area6, 7: area7, 8: area8, 9: area9, 10: area10, 11: area11, 12: area12}

            area_max = np.nansum(area_dict.get(max_sector, None))
            area_ratio = []
            for i in range(1,13):
                area_int = np.nansum(area_dict.get(i, None))
                area_ratio.append((area_int/area_max)*100)


    if edge_selection:
        if manual_edge_detection_checkbox: 
            if not st.session_state.submit_edge:
    
                contours  = skimage.measure.find_contours(smap_mask.data,level =0.99)
                limb = mp.find_limb(smap_mask)
                C = ccc.cartesian


                fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 8), subplot_kw={'projection':smap_mask})
                smap_mask.plot(annotate=False,axes=ax,alpha=0.3,cmap='binary')
                smap_mask.draw_limb(color = 'r')


                ax.plot_coord(ccc, 'o', color = 'black',markersize=10) # center of the flare
                overlay = ax.get_coords_overlay(new_frame)
                overlay[0].set_ticks(np.arange(start = -15,stop = 345,step = 30) * u.deg)
                overlay[0].grid(ls='-', color='blue',linewidth=2)
                plt.draw()

                plt.close()

                x_pix=[]
                y_pix=[]

                for k in range(len(overlay[0].__dict__['grid_lines'])):
                    grid_lines = overlay[0].__dict__['grid_lines'][k]
                    ee = [grid_lines.vertices[i][0] for i in range(len(grid_lines)) if grid_lines.codes[i] == 2]
                    ff = [grid_lines.vertices[i][1] for i in range(len(grid_lines)) if grid_lines.codes[i] == 2]

                    x_pix.append(np.array(ee))
                    y_pix.append(np.array(ff))



                fig = go.Figure()

        # Add each contour as a separate scatter trace
                for i, contour in enumerate(contours):
                    fig.add_trace(go.Scatter(
                        x=contour[:, 1], 
                        y=contour[:, 0],
                        mode='lines',
                        line=dict(width=1, color='red'),
                        showlegend=False 
                    ))

                for i in range(len(limb)):
                    fig.add_trace(go.Scatter(
                x=[smap_mask.world_to_pixel(limb[i])[0].to_value()],  # X-coordinate in pixels
                y=[smap_mask.world_to_pixel(limb[i])[1].to_value()],  # Y-coordinate in pixels
                mode='markers',
                marker=dict(
                    size=5,
                    color='green',
                    symbol='circle'
                    
                )
                ))
                    
                fig.add_trace(go.Scatter(
                x=[smap_mask.world_to_pixel(ccc)[0].to_value()],  # X-coordinate in pixels
                y=[smap_mask.world_to_pixel(ccc)[1].to_value()],  # Y-coordinate in pixels
                mode='markers',
                marker=dict(
                size=10,
                color='blue',
                symbol='circle'
            
                )
                ))



                for i in range(len(x_pix)):
                    fig.add_trace(go.Scatter(
                    x=x_pix[i],  # X-coordinate in pixels
                    y=y_pix[i],  # Y-coordinate in pixels
                    mode='markers',
                    marker=dict(
                        size=1,
                        color='blue',
                        symbol='circle'
                        
                    )
                    ))

                fig.update_yaxes(
                    scaleanchor="x",
                    scaleratio=1,
                )


                # Update layout
                fig.update_layout(
                    title='Dimming Edge selection',
                    xaxis_title='X-axis',
                    yaxis_title='Y-axis',
                    hovermode='closest',
                    width=800,  # Fixed width in pixels
                    height=800,
                    showlegend=False  # Fixed height in pixels
                )

                # Display in Streamlit
                st.plotly_chart(fig)

                with st.form("edge_selection_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        edge1_x = st.number_input("Edge 1 - X coordinate (in pixels)", 
                                                value=st.session_state.edge_coords['edge1_x'],
                                                format="%0.1f")
                        edge2_x = st.number_input("Edge 2 - X coordinate (in pixels)", 
                                                value=st.session_state.edge_coords['edge2_x'],
                                                format="%0.1f")
                    with col2:
                        edge1_y = st.number_input("Edge 1 - Y coordinate (in pixels)", 
                                                value=st.session_state.edge_coords['edge1_y'],
                                                format="%0.1f")
                        edge2_y = st.number_input("Edge 2 - Y coordinate (in pixels)", 
                                                value=st.session_state.edge_coords['edge2_y'],
                                                format="%0.1f")
                    
                    submitted = st.form_submit_button("Submit Edge Coordinates")
                    
                    if submitted:
                        st.session_state.submit_edge = True
                        st.session_state.edge_coords = {
                        'edge1_x': edge1_x,
                        'edge1_y': edge1_y,
                        'edge2_x': edge2_x,
                        'edge2_y': edge2_y
                        }
                    
                        st.write('submit')
                        dimming_edge_world = smap_mask.pixel_to_world(
                        st.session_state.edge_coords['edge1_x'] * u.pix, 
                        st.session_state.edge_coords['edge1_y'] * u.pix
                        )
                        st.write('maybe')
                        dimming_edge_world_2 = smap_mask.pixel_to_world(
                        st.session_state.edge_coords['edge2_x'] * u.pix, 
                        st.session_state.edge_coords['edge2_y'] * u.pix
                        )
                        st.write('maybe_2')
            else:
                pass
                
                
                    
                    
        
        else:
            
            
        ##### Find edge 1

            with st.spinner(f"Finding Edges:",show_time=True):
                try:
                    dist_source_edge_1_all=[]
                    dimming_edge_world_HSH_all=[]
                    deg=[]
                    

                    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 8), subplot_kw={'projection':smap_mask})
                    smap_mask.plot(annotate=False,axes=ax,alpha=0.3,cmap='binary')
                    smap_mask.draw_limb(color = 'r')


                    ax.plot_coord(ccc, 'o', color = 'black',markersize=10) # center of the flare
                    overlay = ax.get_coords_overlay(new_frame)
                    overlay[0].set_ticks(np.arange(start = -15,stop = 345,step = 30) * u.deg)
                    overlay[0].grid(ls='-', color='blue',linewidth=2)
                    start_deg_dict = {1: -180, 2: -150, 3: -120, 4: -90, 5: -60, 6: -30, 7: 0, 8: 30, 9: 60, 10: 90, 11: 120, 12: 150}
                    start_deg = start_deg_dict[max_sector]

                    iter_one = 6
                    for i in range(start_deg-15,start_deg+16,iter_one):
                        overlay_2 = ax.get_coords_overlay(new_frame)
                        overlay_2[0].set_ticks(np.arange(start = i,stop = i+1,step = 30) * u.deg)
                        overlay_2[0].grid(ls='--', color='pink',linewidth=2)
                        deg.append(i)
                        plt.draw()

                    # To find dimming edge automatically
                        grid_lines = overlay_2[0].__dict__['grid_lines'][0]
                        ee = [grid_lines.vertices[i][0] for i in range(len(grid_lines)) if grid_lines.codes[i] == 2]
                        ff = [grid_lines.vertices[i][1] for i in range(len(grid_lines)) if grid_lines.codes[i] == 2]
                        
                        ee = np.array(ee)
                        ff = np.array(ff)

                    # Ensure that ee and ff are integers for indexing
                        ee_int = ee.astype(int)
                        ff_int = ff.astype(int)
                        
                        # Use boolean indexing to find where smap_mask.data is 1 at the given indices
                        mask = smap_mask.data[ff_int, ee_int] == 1
                        dimming_edge_pix = np.column_stack((ee[mask], ff[mask]))
                        if len(dimming_edge_pix)==0:
                            break
                        else:
                            dimming_edge_world = smap_mask.pixel_to_world(dimming_edge_pix[0][0] * u.pix, dimming_edge_pix[0][1] * u.pix)
                            dimming_edge_world_HSH = (dimming_edge_world.transform_to(HeliographicStonyhurst))
                            dimming_edge_world_HSH_all.extend([dimming_edge_world_HSH])
                            edge_1 = dimming_edge_world_HSH.cartesian.xyz.to_value()
                            source_region = ccc.cartesian.xyz.to_value()
                            dist_source_edge = np.linalg.norm(np.subtract(edge_1, source_region)) 
                            dist_source_edge_1_all.extend([dist_source_edge])
                    temp_deg_1 = deg[np.argmax(dist_source_edge_1_all)] 

                    temp_deg_dict = [15, 45, 75, 105, 135, 165, -15, -45, -75, -105, -135, -165]
                    dist_source_edge_1_all=[]
                    dimming_edge_world_HSH_all=[]
                    deg=[]
                    iter_two=2
                    #overlay_2 = ax.get_coords_overlay(new_frame)
                    for i in range(temp_deg_1-iter_one,temp_deg_1+iter_one+1,iter_two):
                        
                        overlay_2[0].set_ticks(np.arange(start = i,stop = i+1,step = 30) * u.deg)
                        overlay_2[0].grid(ls='--', color='red',linewidth=2)
                        deg.append(i)
                        plt.draw()

                    # To find dimming edge automatically
                        grid_lines = overlay_2[0].__dict__['grid_lines'][0]
                        ee = [grid_lines.vertices[i][0] for i in range(len(grid_lines)) if grid_lines.codes[i] == 2]
                        ff = [grid_lines.vertices[i][1] for i in range(len(grid_lines)) if grid_lines.codes[i] == 2]
                        
                        ee = np.array(ee)
                        ff = np.array(ff)

                    # Ensure that ee and ff are integers for indexing
                        ee_int = ee.astype(int)
                        ff_int = ff.astype(int)
                        
                        # Use boolean indexing to find where smap_mask.data is 1 at the given indices
                        mask = smap_mask.data[ff_int, ee_int] == 1
                        dimming_edge_pix = np.column_stack((ee[mask], ff[mask]))
                        dimming_edge_world = smap_mask.pixel_to_world(dimming_edge_pix[0][0] * u.pix, dimming_edge_pix[0][1] * u.pix)
                        dimming_edge_world_HSH = (dimming_edge_world.transform_to(HeliographicStonyhurst))
                        dimming_edge_world_HSH_all.extend([dimming_edge_world_HSH])
                        edge_1 = dimming_edge_world_HSH.cartesian.xyz.to_value()
                        source_region = ccc.cartesian.xyz.to_value()
                        dist_source_edge = np.linalg.norm(np.subtract(edge_1, source_region)) 
                        dist_source_edge_1_all.extend([dist_source_edge])
                    temp_deg = deg[np.argmax(dist_source_edge_1_all)]    
                    dimming_edge_world = dimming_edge_world_HSH_all[np.argmax(dist_source_edge_1_all)]
                    dominant_dir  = np.arange(temp_deg_1-iter_one,temp_deg_1+iter_one+1,iter_two)[np.argmax(dist_source_edge_1_all)]
                    dimming_edge_world_2 = ccc
                    dimming_edge_pix_2 = smap_mask.world_to_pixel(ccc)

                    ax.plot_coord(dimming_edge_world, 'o', color = 'black', markersize=1)

                    st.success('Edge 1 found')
                except Exception as e:
                    st.error('Edge 1 not found')
                    st.error(str(e))
                    st.write('Script has stopped')
                    st.stop()

                #Find edge 2

                try:


                    start_deg_dict = {1: -180, 2: -150, 3: -120, 4: -90, 5: -60, 6: -30, 7: 0, 8: 30, 9: 60, 10: 90, 11: 120, 12: 150}

                    L = [1, 2, 3, 4,5,6,7,8,9,10,11,12]

                    cycled = cycle(L)  # cycle thorugh the list 'L'
                    skipped = dropwhile(lambda x: x != max_sector, cycled)  # drop the values until x==4
                    sliced = islice(skipped, None, 12)  # take the first 12 values
                    result = list(sliced)  # create a list from iterator
                    new_sector = result[3:10] #4:9
                    dimming_edge_world_2_temp=[]
                    dist_edge_temp=[]

                    source_region = ccc.cartesian.xyz.to_value()
                    for k in range(len(new_sector)):
                        if area_ratio[new_sector[k]-1]!= 0:
                            dist_source_edge_2_all=[]
                            dimming_edge_world_HSH_all_2=[]
                            
                            new_deg = start_deg_dict[new_sector[k]]
                            
                            
                            #skip three sectors
                            overlay_2 = ax.get_coords_overlay(new_frame)
                            for i in range(new_deg-15,new_deg+16,3):
                                
                                overlay_2[0].set_ticks(np.arange(start = i,stop = i+1,step = 30) * u.deg)
                                overlay_2[0].grid(ls='--', color='blue',linewidth=2)
                                plt.draw()
                            
                            # To find dimming edge automatically
                                grid_lines = overlay_2[0].__dict__['grid_lines'][0]
                                ee = [grid_lines.vertices[i][0] for i in range(len(grid_lines)) if grid_lines.codes[i] == 2]
                                ff = [grid_lines.vertices[i][1] for i in range(len(grid_lines)) if grid_lines.codes[i] == 2]
                                
                                #dimming_edge_pix = []
                                # ee_set = set(ee)
                                # ff_set = set(ff)
                                # for i in range(len(ee)):
                                #     if smap_mask.data[int(ff[i])][int(ee[i])] == 1:
                                #         dimming_edge_pix.append([ee[i], ff[i]])
                                ee = np.array(ee)
                                ff = np.array(ff)
                        
                        # Ensure that ee and ff are integers for indexing
                                ee_int = ee.astype(int)
                                ff_int = ff.astype(int)
                            
                                # Use boolean indexing to find where smap_mask.data is 1 at the given indices
                                mask = smap_mask.data[ff_int, ee_int] == 1
                                
                                dimming_edge_pix = np.column_stack((ee[mask], ff[mask]))
                                #if 'dimming_edge_pix' not in locals():
                                if len(dimming_edge_pix)==0:
                                    continue
                                else:
                                    dimming_edge_world_2 = smap_mask.pixel_to_world(dimming_edge_pix[0][0] * u.pix, dimming_edge_pix[0][1] * u.pix)
                                    dimming_edge_world_HSH = (dimming_edge_world_2.transform_to(HeliographicStonyhurst))
                                    dimming_edge_world_HSH_all_2.append(dimming_edge_world_HSH)
                                    edge_2 = dimming_edge_world_HSH.cartesian.xyz.to_value()
                                    
                                    dist_source_edge = np.linalg.norm(np.subtract(edge_2, source_region)) 
                                    dist_source_edge_2_all.append(dist_source_edge)
                        
                                dimming_edge_world_2_temp.append(dimming_edge_world_HSH_all_2[np.argmax(dist_source_edge_2_all)])
                                dist_edge_temp.append(np.max(dist_source_edge_2_all))
            
                    dimming_edge_world_3 = dimming_edge_world_2_temp[np.argmax(dist_edge_temp)]
                    overlay_3 = ax.get_coords_overlay(new_frame)
                    overlay_3[0].set_ticks(np.arange(start = temp_deg-180,stop = temp_deg-180+1,step = 30) * u.deg)
                    overlay_3[0].grid(ls='--', color='blue',linewidth=2)
                    plt.draw()    

                    # To find dimming edge automatically
                    grid_lines = overlay_3[0].__dict__['grid_lines'][0]
                    ee_2 = [grid_lines.vertices[i][0] for i in range(len(grid_lines)) if grid_lines.codes[i] == 2]
                    ff_2 = [grid_lines.vertices[i][1] for i in range(len(grid_lines)) if grid_lines.codes[i] == 2]
                    dimming_edge_pix_2 = []
                    ee_2_set = set(ee_2)
                    ff_2_set = set(ff_2)
                    edge_4 = dimming_edge_world_3.cartesian.xyz.to_value()
                    for i in range(0,len(ee_2),5):
                            
                            dimming_edge_world_2 = smap_mask.pixel_to_world(ee_2[i] * u.pix, ff_2[i] * u.pix)
                            dimming_edge_world_2 = (dimming_edge_world_2.transform_to(HeliographicStonyhurst))
                            edge_3 = dimming_edge_world_2.cartesian.xyz.to_value()
                            dist_source_edge = np.linalg.norm(np.subtract(edge_3, source_region)) 
                            #dist_source_edge = np.linalg.norm(np.subtract(edge_3, edge_4)) 
                            if dist_source_edge < np.max(dist_edge_temp):
                                break
                    
                    st.success('Edge 2 found')
                    generate_cones = True
                except:
                    pass
                    st.write('No edge 2 found. Taking Flare source as edge 2')
                    dimming_edge_pix_2 = smap_mask.world_to_pixel(ccc)
                    generate_cones = True
                            

    


                    
    if generate_cones or st.session_state.submit_edge:
        with st.spinner(f"Plotting Edges:",show_time=True):
            xlims_world = [-smap_mask.data.shape[0]-150, smap_mask.data.shape[0]+150]*u.arcsec
            ylims_world = [-smap_mask.data.shape[1]-150, smap_mask.data.shape[1]+150]*u.arcsec

            world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world, frame=smap_mask.coordinate_frame)
            pixel_coords_x, pixel_coords_y = smap_mask.wcs.world_to_pixel(world_coords)
            fig_edge, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(9, 8), subplot_kw={'projection':smap_mask})

            smap_mask.plot(annotate=False,axes=ax,alpha=0.3,cmap='binary')
            smap_mask.draw_limb(color='red')
            overlay = ax.get_coords_overlay(new_frame)
            yy = np.arange(start = -15,stop = 345,step = 30)
            yy = np.array([-15, 15, 45,  75, 105, 135, 165, 195, 225, 255, 285, 315])
            overlay[0].set_ticks(yy * u.deg)
            #overlay[0].set_ticks(spacing=30. * u.deg, color='white')
            overlay[0].grid(ls='-', color='blue',linewidth=2)
            overlay[0].set_axislabel('Latitude', minpad=1)
            overlay[1].set_axislabel('Longitude', minpad=1)
            overlay = ax.get_coords_overlay(new_frame)
            try:
                yy = np.array([dominant_dir])
                overlay[0].set_ticks(yy * u.deg)
                overlay[0].grid(ls='--', color='indigo',linewidth=1.5)
                ax.plot(1,1, ls='--', color='indigo',linewidth=1,label='Dominant dimming direction')
            except:
                pass
            contours  = skimage.measure.find_contours(smap_mask.data,level =0.99)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.8,color ='r',label='Dimming contour')

            ax.text(0.5, 0.92, "1",fontsize=20, transform=ax.transAxes)
            ax.text(0.2, 0.82, "2",fontsize=20, transform=ax.transAxes)
            ax.text(0.07, 0.65, "3",fontsize=20, transform=ax.transAxes)
            ax.text(0.05, 0.45, "4",fontsize=20, transform=ax.transAxes)
            ax.text(0.08, 0.28, "5",fontsize=20, transform=ax.transAxes)
            ax.text(0.3, 0.05, "6",fontsize=20, transform=ax.transAxes)
            ax.text(0.56, 0.03, "7",fontsize=20, transform=ax.transAxes)
            ax.text(0.75, 0.23, "8",fontsize=20, transform=ax.transAxes)
            ax.text(0.92, 0.33, "9",fontsize=20, transform=ax.transAxes)
            ax.text(0.92, 0.54, "10",fontsize=20, transform=ax.transAxes)
            ax.text(0.88, 0.68, "11",fontsize=20, transform=ax.transAxes)
            ax.text(0.75, 0.84, "12",fontsize=20, transform=ax.transAxes)


            plt.tick_params(
                axis='x',          
                which='both',     
                top=False, bottom=True, direction ='in'
            ) 
            plt.tick_params(
                axis='y',  
                which='both',      
                right=False, left=True, direction ='in'
            ) 

            plt.xlabel('X (Arcsec)',fontsize=12)
            plt.ylabel('Y (Arcsec)',fontsize=12)
            ax.set_xlim(pixel_coords_x)
            ax.set_ylim(pixel_coords_y)

            lon = ax.coords[0]
            lat = ax.coords[1]
            lon.set_ticks_visible(False)
            lon.set_ticklabel_visible(False)
            lat.set_ticks_visible(False)
            lat.set_ticklabel_visible(False)
            lon.set_axislabel('')

            ax.coords.frame.set_color('white')



            handles, labels = ax.get_legend_handles_labels()


            by_label = dict(zip(labels, handles))
            fig_edge.legend(by_label.values(), by_label.keys(),ncol=3,markerscale=1, bbox_transform=fig_edge.transFigure,handletextpad=0.2, labelspacing = 0.4 , borderpad=0.2,prop={'size': 16}, bbox_to_anchor=(0.4, 0.10),loc='center',frameon=False)




            ## AXIN

            axin = ax.inset_axes([1.05, 0.01, 0.6, 1.2],projection=smap_mask)
            smap_mask.plot(annotate=False,axes=axin,alpha=0.3,cmap='binary')

            smap_mask.draw_limb(axes=axin,color='red')
            xpix = smap_mask.world_to_pixel(dimming_edge_world)[0].to_value()
            ypix = smap_mask.world_to_pixel(dimming_edge_world)[1].to_value()
            xpix_2 = smap_mask.world_to_pixel(dimming_edge_world_2)[0].to_value()
            ypix_2 = smap_mask.world_to_pixel(dimming_edge_world_2)[1].to_value()
            axin.set_xlim(min(xpix,xpix_2)-75, max(xpix,xpix_2)+75)
            axin.set_ylim(min(ypix,ypix_2)-75, max(ypix,ypix_2)+75)
            
            overlay2 = axin.get_coords_overlay(new_frame)

            yy = np.arange(start = -15,stop = 345,step = 30)
            yy = np.array([-15, 15, 45,  75, 105, 135, 165, 195, 225, 255, 285, 315])
            overlay2[0].set_ticks(yy * u.deg)
            #overlay[0].set_ticks(spacing=30. * u.deg, color='white')
            overlay2[0].grid(ls='-', color='blue',linewidth=2)
            overlay2[0].set_axislabel('', minpad=1)
            overlay2[1].set_axislabel('', minpad=1)

            overlay2[0].set_ticks(color='white')
            overlay2[1].set_ticks(color='white')


            overlay2[0].set_ticklabel(color='white',exclude_overlapping=True)
            overlay2[1].set_ticklabel(color='white',exclude_overlapping=True)
            overlay2[0].set_axislabel(' ', minpad=1)
            overlay2[1].set_axislabel(' ', minpad=1)

            overlay3 = axin.get_coords_overlay(new_frame)
            try:
                yy = np.array([dominant_dir])
                overlay3[0].set_ticks(yy * u.deg)
            #overlay[0].set_ticks(spacing=30. * u.deg, color='white')
                overlay3[0].grid(ls='--', color='indigo',linewidth=1.5)
                overlay3[0].set_axislabel('', minpad=1)
                overlay3[1].set_axislabel('', minpad=1)

                overlay3[0].set_ticks(color='white')
                overlay3[1].set_ticks(color='white')
            except:
                pass

            for contour in contours:
                axin.plot(contour[:, 1], contour[:, 0], linewidth=0.8,color ='r')

            overlay3[0].set_ticklabel(color='white',exclude_overlapping=True)
            overlay3[1].set_ticklabel(color='white',exclude_overlapping=True)
            overlay3[0].set_axislabel(' ', minpad=1)
            overlay3[1].set_axislabel(' ', minpad=1)

            axin.plot_coord(ccc, 'o', color = 'black',markersize=10) # center of the flare
            axin.plot_coord(dimming_edge_world, 'o', color = 'green', markersize=7)
            ax.plot_coord(ccc, 'o', color = 'black',markersize=10) # center of the flare
            ax.plot_coord(dimming_edge_world, 'o', color = 'green', markersize=7)
            ax.plot_coord(dimming_edge_world_2, 'o', color = 'green', markersize=7)
            axin.plot_coord(dimming_edge_world_2, 'o', color = 'green', markersize=7)

            lon = axin.coords[0]
            lat = axin.coords[1]
            lon.set_ticks_visible(False)
            lon.set_ticklabel_visible(False)
            lat.set_ticks_visible(False)
            lat.set_ticklabel_visible(False)
            lon.set_axislabel('')
            lat.set_axislabel('')
            axin.grid('off')
            axin.coords.frame.set_color('silver')

            ax.indicate_inset_zoom(axin)
    

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1,dpi=300)
            plt.close(fig_edge)  # Important: close the figure to free memory
            st.image(buf)
        
            
            if save_all_plots_checkbox:
                fig_edge.savefig(os.path.join(save_path_plots, 'dimming_edges'+'.png'),facecolor='w',bbox_inches='tight',dpi=300)
                st.success(f'Plot saved in {save_path_plots} as dimming_edges.png')





    ### Make cones

        with st.spinner(f"Generating Cones:",show_time=True):

            x_sphere,y_sphere,z_sphere = direcd.create_sphere(cx=0,cy=0,cz=0,r = Rsun)


            cen_pix = smap_mask.world_to_pixel(center)
            source = smap_mask.world_to_pixel(ccc)
            sr_pix = [(smap_mask.world_to_pixel(dimming_edge_world_2)[0]),(smap_mask.world_to_pixel(dimming_edge_world_2)[1])]
            far_pix=[round(smap_mask.world_to_pixel(dimming_edge_world)[0].to_value()),round(smap_mask.world_to_pixel(dimming_edge_world)[1].to_value())]
            OA_x = np.linspace(cen_pix[0].to_value(),sr_pix[0].to_value(),100)
            OA_y = np.linspace(cen_pix[1].to_value(),sr_pix[1].to_value(),100)
            OB_x = np.linspace(cen_pix[0].to_value(),far_pix[0],100)
            OB_y = np.linspace(cen_pix[1].to_value(),far_pix[1],100)
            B_world = smap_mask.pixel_to_world((OB_x[-1])*u.pix, (OB_y[-1])*u.pix)
            A_world = smap_mask.pixel_to_world(OA_x[-1]*u.pix, OA_y[-1]*u.pix)
            A_cart = A_world.transform_to(HeliographicStonyhurst).cartesian
            B_cart = B_world.transform_to(HeliographicStonyhurst).cartesian


            #


            if np.any(np.isnan(B_cart.xyz.to_value())) ==True:
                with Helioprojective.assume_spherical_screen(smap_mask.observer_coordinate):
                    B_cart = B_world.transform_to(HeliographicStonyhurst).cartesian
                    print('on limb')

            end_height = 3
            points_array = end_height * 500
            OB_3D, OA_3D = B_cart.xyz.value, A_cart.xyz.value
            unit_slope_OB = OB_3D / np.linalg.norm(OB_3D)
            unit_slope_OA = OA_3D / np.linalg.norm(OA_3D)
            OB_extend = OB_3D + (unit_slope_OB* end_height*Rsun)
            OB_points = np.linspace(OB_3D,OB_extend,points_array)
            OA_extend = OA_3D + (unit_slope_OA* end_height*Rsun)
            OA_points = np.linspace(OA_3D,OA_extend,points_array)
            cen_line = (OA_points + OB_points)/2

            dist=[]
            dist_2=[]
            index_height_OA=[]
            index_height_OB=[]
            range_count=[]
            step = 0.1
            C = ccc.cartesian
            for i in range(len(OA_points)):
                dist.append(np.sqrt((C.xyz.value[0] - OA_points[i,0])**2 + (C.xyz.value[1] - OA_points[i,1])**2 + (C.xyz.value[2] - OA_points[i,2])**2)/Rsun)
                dist_2.append(np.sqrt((C.xyz.value[0] - OB_points[i,0])**2 + (C.xyz.value[1] - OB_points[i,1])**2 + (C.xyz.value[2] - OB_points[i,2])**2)/Rsun)

            dist_OA = np.round(dist,2)
            dist_OB = np.round(dist_2,2) 

           
            range_num = np.arange(0.1,end_height+step,step)

            for i in range(len(range_num)):
                
                if round(range_num[i],2) in dist_OA:
                    if round(range_num[i],2) in dist_OB:

                        index_height_OA.append(list(dist_OA).index(round(range_num[i],2)))
                        index_height_OB.append(list(dist_OB).index(round(range_num[i],2)))
                        range_count.append(round(range_num[i],2))
            

            cen_3D_x_2 = [np.linspace(OA_points[idx_OA, 0], OB_points[idx_OB, 0], 1000) for idx_OA, idx_OB in zip(index_height_OA, index_height_OB)]
            cen_3D_y_2 = [np.linspace(OA_points[idx_OA, 1], OB_points[idx_OB, 1], 1000) for idx_OA, idx_OB in zip(index_height_OA, index_height_OB)]
            cen_3D_z_2 = [np.linspace(OA_points[idx_OA, 2], OB_points[idx_OB, 2], 1000) for idx_OA, idx_OB in zip(index_height_OA, index_height_OB)]
            mid_x = [(line[0] + line[-1]) / 2 for line in cen_3D_x_2]
            mid_y = [(line[0] + line[-1]) / 2 for line in cen_3D_y_2]
            mid_z = [(line[0] + line[-1]) / 2 for line in cen_3D_z_2]

            C = ccc.cartesian

            width=[]
            for i in range(len(index_height_OA)):
                vec2 = C.xyz.value - np.array([cen_3D_x_2[i][-1],cen_3D_y_2[i][-1],cen_3D_z_2[i][-1]])
                vec1 = C.xyz.value - np.array([cen_3D_x_2[i][0],cen_3D_y_2[i][0],cen_3D_z_2[i][0]])
                width.append((np.degrees(np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))))

            angle_2=[]
            cone_x=[]
            cone_y=[]
            cone_z=[]

            P_x=[]
            P_y=[]
            P_z=[]

            cone_lat=[]
            cone_lon=[]

            X3_array=[]
            Y3_array=[]
            Z3_array=[]
            height_cone=[]
            c=[]
            j=0

            
            for iter in range(len(range_count)):

            #for iter in range(0,5):


                h = ((np.sqrt((mid_x[iter] - C.x/u.m)**2 + (mid_y[iter] - C.y/u.m)**2 + (mid_z[iter] - C.z/u.m)**2)))
                
                height_cone.append(h.to_value())

                Bubble_x = mid_x[iter]
                Bubble_y = mid_y[iter]
                Bubble_z = mid_z[iter]


                P_x.append(Bubble_x)
                P_y.append(Bubble_y)
                P_z.append(Bubble_z)

                vec1 = np.array([0,0,0]) - C.xyz.value
                vec2 = C.xyz.value - np.array([Bubble_x,Bubble_y,Bubble_z])
                
                angle_2.append((np.degrees(np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))))

                theta = width[iter]/2
                r = h*math.tan(np.pi*theta/180);
                m = h/r;
                ra = np.linspace(0,r,11) ;
                th = np.linspace(0,2*np.pi,41);

                [Ra,A] = np.meshgrid(ra,th);
            # Generate cone about Z axis with given aperture angle and height
                X =  Ra* np.cos(A);
                Y =  Ra* np.sin(A);
                Z =  m*Ra;
                #% Cone around the z-axis, point at the origin
                #% find coefficients of the axis vector xi + yj + zk
                xx = (C.x/u.m - P_x[j]).to_value(u.dimensionless_unscaled)
                yy = (C.y/u.m- P_y[j]).to_value(u.dimensionless_unscaled)
                zz = (C.z/u.m - P_z[j]).to_value(u.dimensionless_unscaled)

            # % find angle made by axis vector with X axis
                phix = math.atan2(yy,xx);
                #phix = 0
                #% find angle made by axis vector with Z axis
                phiz = math.atan2(np.sqrt(xx**2 + yy**2),(zz));
                #phiz = (math.pi)/2

                #% Rotate once about Z axis 
                X1 = (X*np.cos(phiz)+Z*np.sin(phiz));
                Y1 = (Y);
                Z1 = (-X*np.sin(phiz)+Z*np.cos(phiz));
                #% Rotate about X axis
                X3 = (C.x/u.m - (X1*np.cos(phix)-Y1*np.sin(phix)));
                Y3 = (C.y/u.m - (X1*np.sin(phix)+Y1*np.cos(phix)));
                Z3 = (C.z/u.m -  (Z1));



                X3_array.append(X3)
                Y3_array.append(Y3)
                Z3_array.append(Z3)

                    
                Z_int = Z3.reshape(-1, 1)
                Y_int = Y3.reshape(-1, 1)
                X_int = X3.reshape(-1, 1)

            # Compute transformed coordinates
                X_P = np.array((Rsun * X_int) / np.sqrt(X_int**2 + Y_int**2 + Z_int**2))
                Y_P = np.array((Rsun * Y_int) / np.sqrt(X_int**2 + Y_int**2 + Z_int**2))
                Z_P = np.array((Rsun * Z_int) / np.sqrt(X_int**2 + Y_int**2 + Z_int**2))
            # Compute latitudes and longitudes
                lat = np.array(np.degrees(np.arcsin(Z_P / Rsun)))
                lon = np.array(np.degrees(np.arcsin(Y_P / (Rsun * np.cos(np.arcsin(Z_P / Rsun))))))

                cone_x.append(X_P)
                cone_y.append(Y_P)
                cone_z.append(Z_P)

                cone_lat.append(lat)
                cone_lon.append(lon)

                j = j+1;
                
                


            for i in range(len(cone_lat)):
                

                c.append(SkyCoord(cone_lon[i]*u.deg,cone_lat[i]*u.deg, 695700*u.km,
                        frame="heliographic_stonyhurst",
                        obstime=smap.meta['date-obs'], observer='earth'))

                    
            # Compute transformed coordinates
            xe = np.array(P_x) * (Rsun / np.linalg.norm([P_x, P_y, P_z], axis=0))
            ye = np.array(P_y) * (Rsun / np.linalg.norm([P_x, P_y, P_z], axis=0))
            ze = np.array(P_z) * (Rsun / np.linalg.norm([P_x, P_y, P_z], axis=0))

            # Compute distances
            dist_height = np.linalg.norm([np.array(P_x) - xe, np.array(P_y) - ye, np.array(P_z) - ze], axis=0) / Rsun



            
            try:
                count = len(dist_height)
                for height_del in range(count-1):
                    if dist_height[height_del] > dist_height[height_del+1]:
                        width.pop(height_del)
                        angle_2.pop(height_del)
                        range_num = np.delete(range_num, height_del)
                        index_height_OA.pop(height_del)
                        index_height_OB.pop(height_del)
                        range_count.pop(height_del)
                        cen_3D_x_2.pop(height_del)
                        cen_3D_y_2.pop(height_del)
                        cen_3D_z_2.pop(height_del)
                        mid_x.pop(height_del)
                        mid_y.pop(height_del)
                        mid_z.pop(height_del)
                        height_cone.pop(height_del)
                        P_x.pop(height_del)
                        P_y.pop(height_del)
                        P_z.pop(height_del)
                        X3_array.pop(height_del)
                        Y3_array.pop(height_del)
                        Z3_array.pop(height_del)
                        cone_x.pop(height_del)
                        cone_y.pop(height_del)
                        cone_z.pop(height_del)
                        cone_lat.pop(height_del)
                        cone_lon.pop(height_del)
                        c.pop(height_del)
                delete_indices = np.where(dist_height[:-1] > dist_height[1:])[0]
                dist_height = np.delete(dist_height, delete_indices)
                st.write('done')
            except Exception as e:
                st.write(str(e))
                pass




           

                         

            st.success('Cones Generated')
            
            
        
        if st.session_state.ensemble_cones:
            with st.spinner(f"Plotting Ensemble of Cones:",show_time=True):
                
                try:
                    Q_2 = SkyCoord(CartesianRepresentation(C.x,C.y,C.z),frame=HeliographicStonyhurst,
                            obstime='7-Jun-2021 04:52:00.620', observer=smap.coordinate_frame.observer)
                    Q_2 = Q_2.transform_to(HeliographicStonyhurst)

                    north_2=Q_2
                    tt, kk = smap_mask.world_to_pixel(Q_2)

                    ll = smap_mask.pixel_to_world(tt, kk)


                    new_frame_int = NorthOffsetFrame(north=north_2)
                    #h = all_map.transform_to(new_frame)

                    latitudes = np.arange(-90, 91, 1) * u.deg
                    X_cart_array_final = []
                    Y_cart_array_final = []
                    Z_cart_array_final = []

                    hpc_out = HeliographicStonyhurst()

                    for longitude in range(-180, 180, 30):
                        # Create all coordinates at once using vectorization
                        coords = SkyCoord(np.full_like(latitudes, longitude * u.deg), latitudes, frame=new_frame_int)
                        transformed = coords.transform_to(hpc_out)
                        three_d = transformed.make_3d()
                        
                        # Get cartesian coordinates in one operation
                        x = three_d.cartesian.x.value * 1e3
                        y = three_d.cartesian.y.value * 1e3
                        z = three_d.cartesian.z.value * 1e3
                        
                        # Filter positive x values using boolean indexing
                        mask = x > 0
                        X_cart_array_final.append(x[mask])
                        Y_cart_array_final.append(y[mask])
                        Z_cart_array_final.append(z[mask])
                    CP_x=[]
                    CP_y=[]
                    CP_z=[]
                    count =0
                    contours  = skimage.measure.find_contours(smap_mask.data,level =0.99)


                    coords_dict = {
                        1: [0.2, 0.99],
                        2: [0.69, 0.99],
                        3: [0.2, 0.79],
                        4: [0.69, 0.79],
                        5: [0.2, 0.59],
                        6: [0.69, 0.59],
                        7: [0.2, 0.39],
                        8: [0.69, 0.39],
                        9: [0.2, 0.19],
                        10: [0.69, 0.19],
                        11: [0.2, 0.99],
                        12: [0.69, 0.99],
                        13: [0.2, 0.79],
                        14: [0.69, 0.79],
                        15: [0.2, 0.59],
                        16: [0.69, 0.59],
                        17: [0.2, 0.39],
                        18: [0.69, 0.39],
                        19: [0.2, 0.19],
                        20: [0.69, 0.19]
                        }

                    positions = [(475, 725), (400, 700), (300, 600), (275, 500), 
                                (315, 375), (360, 310), (500, 275), (600, 300), 
                                (675, 375), (725, 475), (700, 575), (600, 700)]
                    # positions = [(525, 850), (325, 800), (175, 650), (150, 450), 
                    #              (200, 300), (325, 200), (525, 150), (725, 250), 
                    #              (825, 375), (850, 575), (775, 700), (700, 775)]
                    labels_sec = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']


                    for i in range(len(P_x)):

                        CP_x.append(np.array([C.x/u.m, P_x[i]]))
                        CP_y.append(np.array([C.y/u.m, P_y[i]]))
                        CP_z.append(np.array([C.z/u.m, P_z[i]]))



                    fig_proj = plt.figure(figsize=(10, 20), num=1, clear=True,layout='tight')

                    range_num = np.arange(0,1,0.1)

                    ay = fig_proj.subplots(5, 5,sharex=True, sharey=True,gridspec_kw=dict(width_ratios=[2,2,0.1,2,2],hspace=0,wspace=0)) # prepare for multiple subplots

                    ay2 = ay.ravel()

                    count_even = 0
                    count_odd = 0
                    iteration=0

                    #for iteration in range(len(range_count)):
                    #for iteration in range(1):    
                        
                    for iteration in range(0,10):
                        
                        
                        if iteration%2==0:
                            count_even = count_even+1
                            count_plot = 5*count_even-4
                        else:
                            count_odd = count_odd+1
                            count_plot = 5*count_odd-1
                            count_space = 5*count_odd-2
                        ay2[count_plot-1].remove()


                        ax = fig_proj.add_subplot(5, 5, count_plot, projection='3d',computed_zorder=False)

                        ax.plot_surface(x_sphere, y_sphere, z_sphere, rstride=3, cstride=3,shade=False,alpha=1,rasterized=True,antialiased=True, color = 'gainsboro',edgecolor='blue',linewidth=0)
                        ax.plot_surface(X3_array[iteration], Y3_array[iteration], Z3_array[iteration], rstride=1, cstride=1,alpha=0.5,color='darkgreen')
                        
                        x2 = np.array([0, C.x/u.m])
                        y2 = np.array([0, C.y/u.m])
                        z2 = np.array([0, C.z/u.m])
                        
                        #ax.plot3D(x2, y2, z2, color='black',linewidth=2)
                        ax.plot3D(CP_x[iteration], CP_y[iteration], CP_z[iteration], color='black',linewidth=2)
                        
                        #ax.text(O.x,O.y,O.z,'O',fontsize=15)
                        ax.text(0,0,0,'O',fontsize=15)
                        ax.text(C.x/u.m,C.y/u.m,C.z/u.m+1e8,'C',fontsize=13)
                        ax.text(P_x[iteration],P_y[iteration],P_z[iteration]+0.5e8,'P',fontsize=13)
                        ax.plot3D(OB_points[:,0], OB_points[:,1], OB_points[:,2], '--',color='black',linewidth=1)
                        ax.plot3D(OA_points[:,0], OA_points[:,1], OA_points[:,2], '--',color='black',linewidth=1)
                    #     ax.plot3D(OB_3D_x_2, OB_3D_y_2, OB_3D_z_2, '--',color='black',linewidth=1)
                    #     ax.plot3D(OA_3D_x_2, OA_3D_y_2, OA_3D_z_2, '--',color='black',linewidth=1)
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.set_zlabel('z')

                        ax.view_init(azim=0,elev=0)
                        ax.axis('off')
                        
                        if iteration in [0,1]:
                            ax.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [2,3]:
                            ax.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [4,5]:
                            ax.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [6,7]:
                            ax.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [8,9]:
                            ax.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [10,11]:
                            ax.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [12,13]:
                            ax.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [14,15]:
                            ax.set_box_aspect([1,1,1], zoom=2)
                        else:
                            ax.set_box_aspect([1,1,1], zoom=2)
                        direcd.set_axes_equal(ax)
                        
                        
                        

                        

                        ay2[count_plot].remove()
                        ax_2 = fig_proj.add_subplot(5, 5, count_plot+1, projection=smap_mask)
                        #ax_2.imshow(smap_mask.data,origin="lower",alpha=0.3,cmap='binary')
                        smap_mask.plot(annotate=False,axes=ax_2,cmap='binary',alpha=0.3)
                        #smap_2.plot(vmin=-1,vmax=1,annotate=False,axes=ax_2)
                        #ax_2.imshow(smap_mask.data,alpha=0.3)
                        smap_mask.draw_limb(color='red')
                            #plt.ioff()
                        ax_2.plot_coord(ccc, 'o', color = 'black',markersize=10) # center of the flare
                        overlay = ax_2.get_coords_overlay(new_frame)
                        overlay[0].set_ticks(np.arange(start = -15,stop = 345,step = 30) * u.deg)
                        overlay[0].grid(ls='-', color='blue',linewidth=2)
                        ax_2.plot_coord(c[iteration], 'o', markersize=1.5,alpha=0.7,label='CME projection')
                        #ax_2.text(0.7, 0.7, "C",fontsize=30, transform=ax_2.transAxes)
                        
                        for contour in contours:
                            ax_2.plot(contour[:, 1], contour[:, 0], linewidth=1,color ='r',label='Dimming contour')

                        

                        for pos, label_sec in zip(positions, labels_sec):
                            ax_2.text(pos[0], pos[1], label_sec, fontsize=5)
                        ax_2.set_xlim(pixel_coords_x)
                        ax_2.set_ylim(pixel_coords_y)
                        lon = ax_2.coords[0]
                        lat = ax_2.coords[1]

                        lon.set_ticks_visible(False)
                        lon.set_ticklabel_visible(False)
                        lat.set_ticks_visible(False)
                        lat.set_ticklabel_visible(False)
                        lon.set_axislabel('')

                        
                        ax_2.coords.frame.set_color('white')
                        
                        handles, labels = ax_2.get_legend_handles_labels()

                        
                        if iteration%2!=0:
                            ay2[count_space-1].remove()
                        count = count + 1
                        
                        
                        txt_coord = coords_dict.get(count, [0, 0])
                        

                        
                        
                        plt.figtext(txt_coord[0],txt_coord[1], 'H = ' + str(round(dist_height[iteration],2)) + '*Rsun' + ', ' + 'W' + ' = ' + str(round(width[iteration],1)) +', ' + r'$\beta$ = '+str(round(angle_2[iteration],1)) + chr(176), ha='center', va='center',fontsize = '10')

                        for x, y, z in zip(X_cart_array_final, Y_cart_array_final, Z_cart_array_final):
                            ax.plot3D(x, y, z, 'blue', zorder=2)
                            
                                    
                            
                            #plt.draw()

                    by_label = dict(zip(labels, handles))
                    fig_proj.tight_layout(pad=1.08)
                    
                    fig_proj.legend(by_label.values(), by_label.keys(),ncol=3,markerscale=2, bbox_transform=fig.transFigure,handletextpad=0.1, labelspacing = 0 , borderpad=0.2,prop={'size': 10}, bbox_to_anchor=(0.5, 0.005),loc='center')
                    # #st.image(fig)
                    buf = BytesIO()
                    fig_proj.savefig(buf, format='png', dpi=300)
                    # Important: close the figure to free memory
                    buf.seek(0)
                    st.image(buf)
                    
                

                    if save_all_plots_checkbox:
                        fig_proj.savefig(os.path.join(save_path_plots, 'projection_1'+'.png'),facecolor='w',bbox_inches='tight',dpi=300)
                        st.success(f'Plot saved in {save_path_plots} as projection_1.png')
                    

                    fig_new = plt.figure(figsize=(10, 20), num=1, clear=True,layout='tight')

                    range_num = np.arange(0,1,0.1)

                    ay = fig_new.subplots(5, 5,sharex=True, sharey=True,gridspec_kw=dict(width_ratios=[2,2,0.1,2,2],hspace=0,wspace=0)) # prepare for multiple subplots

                    ay2 = ay.ravel()

                    count_even = 0
                    count_odd = 0
                    iteration=0

                    #for iteration in range(len(range_count)):
                    #for iteration in range(1):    
                        
                    for iteration in range(10,20):
                        
                        
                        if iteration%2==0:
                            count_even = count_even+1
                            count_plot = 5*count_even-4
                        else:
                            count_odd = count_odd+1
                            count_plot = 5*count_odd-1
                            count_space = 5*count_odd-2
                        ay2[count_plot-1].remove()


                        ax_new = fig_new.add_subplot(5, 5, count_plot, projection='3d',computed_zorder=False)

                        ax_new.plot_surface(x_sphere, y_sphere, z_sphere, rstride=3, cstride=3,shade=False,alpha=1,rasterized=True,antialiased=True, color = 'gainsboro',edgecolor='blue',linewidth=0)
                        ax_new.plot_surface(X3_array[iteration], Y3_array[iteration], Z3_array[iteration], rstride=1, cstride=1,alpha=0.5,color='darkgreen')
                        
                        x2 = np.array([0, C.x/u.m])
                        y2 = np.array([0, C.y/u.m])
                        z2 = np.array([0, C.z/u.m])
                        
                        #ax.plot3D(x2, y2, z2, color='black',linewidth=2)
                        ax_new.plot3D(CP_x[iteration], CP_y[iteration], CP_z[iteration], color='black',linewidth=2)
                        
                        #ax.text(O.x,O.y,O.z,'O',fontsize=15)
                        ax_new.text(0,0,0,'O',fontsize=15)
                        ax_new.text(C.x/u.m,C.y/u.m,C.z/u.m+1e8,'C',fontsize=13)
                        ax_new.text(P_x[iteration],P_y[iteration],P_z[iteration]+0.5e8,'P',fontsize=13)
                        ax_new.plot3D(OB_points[:,0], OB_points[:,1], OB_points[:,2], '--',color='black',linewidth=1)
                        ax_new.plot3D(OA_points[:,0], OA_points[:,1], OA_points[:,2], '--',color='black',linewidth=1)
                    #     ax.plot3D(OB_3D_x_2, OB_3D_y_2, OB_3D_z_2, '--',color='black',linewidth=1)
                    #     ax.plot3D(OA_3D_x_2, OA_3D_y_2, OA_3D_z_2, '--',color='black',linewidth=1)
                        ax_new.set_xlabel('x')
                        ax_new.set_ylabel('y')
                        ax_new.set_zlabel('z')

                        ax_new.view_init(azim=0,elev=0)
                        ax_new.axis('off')
                        
                        if iteration in [0,1]:
                            ax_new.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [2,3]:
                            ax_new.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [4,5]:
                            ax_new.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [6,7]:
                            ax_new.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [8,9]:
                            ax_new.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [10,11]:
                            ax_new.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [12,13]:
                            ax_new.set_box_aspect([1,1,1], zoom=2)
                        elif iteration in [14,15]:
                            ax_new.set_box_aspect([1,1,1], zoom=2)
                        else:
                            ax_new.set_box_aspect([1,1,1], zoom=2)
                        direcd.set_axes_equal(ax_new)
                        
                        
                        

                        
                        ay2[count_plot].remove()
                        ax_2_new = fig_new.add_subplot(5, 5, count_plot+1, projection=smap_mask)
                        #ax_2.imshow(smap_mask.data,origin="lower",alpha=0.3,cmap='binary')
                        smap_mask.plot(annotate=False,axes=ax_2_new,cmap='binary',alpha=0.3)
                        #smap_2.plot(vmin=-1,vmax=1,annotate=False,axes=ax_2)
                        #ax_2.imshow(smap_mask.data,alpha=0.3)
                        smap_mask.draw_limb(color='red')
                            #plt.ioff()
                        ax_2_new.plot_coord(ccc, 'o', color = 'black',markersize=10) # center of the flare
                        overlay = ax_2_new.get_coords_overlay(new_frame)
                        overlay[0].set_ticks(np.arange(start = -15,stop = 345,step = 30) * u.deg)
                        overlay[0].grid(ls='-', color='blue',linewidth=2)
                        ax_2_new.plot_coord(c[iteration], 'o', markersize=1.5,alpha=0.7,label='CME projection')
                        #ax_2.text(0.7, 0.7, "C",fontsize=30, transform=ax_2.transAxes)
                        
                        for contour in contours:
                            ax_2_new.plot(contour[:, 1], contour[:, 0], linewidth=1,color ='r',label='Dimming contour')

                        

                        for pos, label_sec in zip(positions, labels_sec):
                            ax_2_new.text(pos[0], pos[1], label_sec, fontsize=5)
                        ax_2_new.set_xlim(pixel_coords_x)
                        ax_2_new.set_ylim(pixel_coords_y)
                        lon = ax_2_new.coords[0]
                        lat = ax_2_new.coords[1]

                        lon.set_ticks_visible(False)
                        lon.set_ticklabel_visible(False)
                        lat.set_ticks_visible(False)
                        lat.set_ticklabel_visible(False)
                        lon.set_axislabel('')

                        
                        ax_2_new.coords.frame.set_color('white')
                        
                        handles, labels = ax_2_new.get_legend_handles_labels()

                        
                        if iteration%2!=0:
                            ay2[count_space-1].remove()
                        count = count + 1
                        
                        
                        txt_coord = coords_dict.get(count, [0, 0])
                        

                        
                        
                        plt.figtext(txt_coord[0],txt_coord[1], 'H = ' + str(round(dist_height[iteration],2)) + '*Rsun' + ', ' + 'W' + ' = ' + str(round(width[iteration],1)) +', ' + r'$\beta$ = '+str(round(angle_2[iteration],1)) + chr(176), ha='center', va='center',fontsize = '10')

                        
                        for x, y, z in zip(X_cart_array_final, Y_cart_array_final, Z_cart_array_final):
                            ax_new.plot3D(x, y, z, 'blue', zorder=2)

                    by_label = dict(zip(labels, handles))
                    fig_new.tight_layout(pad=1.08)
                    
                    fig_new.legend(by_label.values(), by_label.keys(),ncol=3,markerscale=2, bbox_transform=fig_new.transFigure,handletextpad=0.1, labelspacing = 0 , borderpad=0.2,prop={'size': 10}, bbox_to_anchor=(0.5, 0.005),loc='center')
                    # #st.image(fig)
                    buf = BytesIO()
                    fig_new.savefig(buf, format='png', dpi=300)
                    plt.close(fig_new)  # Important: close the figure to free memory
                    buf.seek(0)
                    st.image(buf)
                    if save_all_plots_checkbox:
                        fig_new.savefig(os.path.join(save_path_plots, 'projection_2'+'.png'),facecolor='w',bbox_inches='tight',dpi=300)
                        st.success(f'Plot saved in {save_path_plots} as projection_2.png')
                    
                    #st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.warning('Cant plot')
                    st.error(str(e))




        try:
            with st.spinner(f"Finding best-fit cone:",show_time=True):

                ppix_y, ppix_x = np.where(smap_mask.data == 1)
                points_dim = np.column_stack((ppix_x, ppix_y))
                mask_boundary = []
                mask_dimming = []
                for row in c:
                    pixel_x, pixel_y = smap_mask.wcs.world_to_pixel(row[10::11])
                    polygon = np.column_stack((pixel_x, pixel_y))

                # Find bounding box
                    left_x, left_y = np.min(polygon, axis=0)
                    right_x, right_y = np.max(polygon, axis=0)
                    
                    # Generate grid points within bounding box
                    x = np.arange(math.ceil(left_x), math.floor(right_x) + 1)
                    y = np.arange(math.ceil(left_y), math.floor(right_y) + 1)

                # Generate points without meshgrid
                    points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
                    path = matplotlib.path.Path(polygon, closed=True)
                    mask = path.contains_points(points)
                    mask_dim = path.contains_points(points_dim)

                    mask_boundary.append(points[mask])
                    mask_dimming.append(points_dim[mask_dim])

                proj_mask=[]
                dim_mask=[]
                for i in range(len(mask_boundary)):
                    data_full_img = np.zeros(smap_mask.data.shape)
                    data_full_img_dim = np.zeros(smap_mask.data.shape)
                    data_full_img[mask_boundary[i][:, 1], mask_boundary[i][:, 0]] = 1
                    data_full_img_dim[mask_dimming[i][:, 1], mask_dimming[i][:, 0]] = 1
                    proj_mask.append(data_full_img)
                    dim_mask.append(data_full_img_dim)

                region_area_dimming = [None] * len(mask_dimming)
                region_area_boundary = [None] * len(mask_boundary)
                for i in range(len(mask_dimming)):

                    region_area_dimming[i] = mp.calculate_region_area(mask_dimming[i], smap_mask)
                    region_area_boundary[i] = mp.calculate_region_area(mask_boundary[i], smap_mask)
                
                region_area_dimming = np.array(region_area_dimming)
                region_area_boundary = np.array(region_area_boundary)
                

                rat_fd = np.flip(np.diff(np.flip(region_area_boundary)))
                rat_fd = np.insert(rat_fd, 0, np.nan)  # Insert NaN for the first element
                    


                five_fd = 0.05 * np.nanmax(np.abs(rat_fd))
                h_fd_idx = np.argmax(rat_fd < five_fd)
                h_fd = dist_height[h_fd_idx]
                a_fd = h_fd_idx

                region_area = np.nansum(area_map.data)
                yy=[]
                for i in range(1,len(dim_mask)):
                    yy.append(((region_area_dimming[i] - region_area_dimming[i-1])/region_area_dimming[i-1])*100)
                yy= np.insert(yy, 0, np.nan)  # Insert NaN for the first element
                rtt = region_area_dimming/region_area_boundary




                height = np.array(dist_height)

                # Plot area
                fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(20, 10),sharex=True)
                ax1.plot(dist_height,rat_fd,'-b',label='Differences in cone projection areas')
                ax1.scatter(dist_height,rat_fd,s=50, color='blue')
                ax1.axhline(y = five_fd,color = 'black', linestyle = '--',linewidth=1)
                #ax1.text(0.1, 0.56, '5% of max ', transform = fig.transFigure,color='blue',fontsize=14)
                ax1.tick_params(axis='x', labelsize=25)
                ax1.axvline(x = h_fd,color = 'k', linestyle = '--',linewidth=2)
                ax1.set_ylabel('Difference in \nProjection Area ($\mathrm{km}^{2}$)', 
                            fontsize=25, 
                            color='blue', 
                            linespacing=1.5)
                ax1.tick_params(axis='y', labelsize=25, labelcolor='blue')
                ax1.yaxis.get_offset_text().set_fontsize(20)
                lines, labels = ax1.get_legend_handles_labels()

                ax2.plot((height), np.round((region_area_dimming/region_area),2)*100, color='green',label='Dimming covered/Total Dimming')
                ax2.scatter((height), np.round((region_area_dimming/region_area),2)*100,s=50, color='green')
                ax2.tick_params(axis='x', labelsize=20)
                ax2.set_xticks(np.arange(0,3,0.2))
                ax2.set_ylim([95,101])
                ax2.set_yticks(np.arange(96,101,2))
                ax2.axvline(x = h_fd,color = 'k', linestyle = '--',linewidth=2)
                ax2.set_ylabel('Percent of dimming covered', fontsize=25,color='green')
                ax2.tick_params(axis='y', labelsize=25,labelcolor='green')
                lines2, labels2 = ax2.get_legend_handles_labels()



                ax2.set_xlabel("Height", fontsize=24)
                ax2.text(0.5, 0.75, 'Best height: '+str(round(dist_height[a_fd],2)) + ' Rsun', transform = fig.transFigure,color='red',fontsize=25)


                ax1.legend(lines, labels, loc=0,fontsize=24,frameon=False)
                ax2.legend(lines2, labels2, loc=0,fontsize=24,frameon=False)
            
                fig.tight_layout()
                plt.subplots_adjust(hspace=.0)

                st.pyplot(fig)

                if save_all_plots_checkbox:
                        fig.savefig(os.path.join(save_path_plots, 'best_height'+'.png'),facecolor='w',bbox_inches='tight',dpi=300)
                        st.success(f'Plot saved in {save_path_plots} as best_height.png')

                
                
                
                
                
                if ((np.abs(np.nanmax(rat_fd/10**8)/np.nanmin(rat_fd/10**8)))<10 or a_fd==2)==True:
                    st.write('hi')
                

                    rat_angle = -np.diff(angle_2)
                    if abs(rat_angle[0]) >100:
                       rat_angle[0] = np.nan

                    rat_angle = np.insert(rat_angle, 0, np.nan)  # Insert NaN for the first element
                    five_percent_dimming_angle = 0.05 * np.nanmax(rat_angle)
                    h_d_idx_angle = np.argmax(rat_angle < five_percent_dimming_angle)
                    h_d_angle = dist_height[h_d_idx_angle]
                    a_d_angle = h_d_idx_angle
                    st.write(a_d_angle)
                    rat_width = -np.diff(width)
                    rat_width = np.insert(rat_width, 0, np.nan)  # Insert NaN for the first element
                    five_percent_dimming_width = 0.05 * np.nanmax(rat_width)
                    h_d_idx_width = np.argmax(rat_width < five_percent_dimming_width)
                    h_d_width = dist_height[h_d_idx_width]
                    a_d_width = h_d_idx_width
                    st.write(a_d_width)
                    a_fd_orig = a_fd
                    a_fd = np.nanmin([a_d_angle,a_d_width])
                    if a_fd==0:
                        a_fd = a_d_angle
                    h_fd = dist_height[a_fd]


                    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(20, 10),sharex=True)
                    ax1.plot(dist_height,rat_angle,'-b',label='Differences in inclination angle')
                    ax1.scatter(dist_height,rat_angle,s=50, color='blue')
                    ax1.axhline(y = five_percent_dimming_angle,color = 'b', linestyle = '--',linewidth=0.5)
                    ax1.axvline(x = h_d_angle,color = 'k', linestyle = '--',linewidth=2)
                    ax1.set_ylabel('Degrees', fontsize=16,color='blue')
                    ax1.tick_params(axis='y', labelsize=15, labelcolor='blue')
                    
                    
                    ax2.plot(dist_height,rat_width, color='green',label='Differences in width')
                    ax2.scatter(dist_height,rat_width,s=50, color='green')
                    ax2.axvline(x = h_d_angle,color = 'k', linestyle = '--',linewidth=2)
                    ax2.set_ylabel('Degrees', fontsize=16,color='green')
                    ax2.tick_params(axis='y', labelsize=15, labelcolor='green')
                    lines, labels = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    
                    ax1.legend(lines, labels, loc=0,fontsize=24,frameon=False)
                    ax2.legend(lines2, labels2, loc=0,fontsize=24,frameon=False)
                    
                    ax2.text(0.5, 0.45, 'Best height: '+str(round(dist_height[a_fd],2)) + ' Rsun', transform = fig.transFigure,color='red',fontsize=18)
                    
                    ax2.set_xlabel("Height", fontsize=24)
                    ax2.tick_params(axis='x', labelsize=20)
                    
                    fig.tight_layout()
                    plt.subplots_adjust(hspace=.0)
                    st.pyplot(fig)

                    if save_all_plots_checkbox:
                        fig.savefig(os.path.join(save_path_plots, 'best_height'+'.png'),facecolor='w',bbox_inches='tight',dpi=300)
                        st.success(f'Updated plot saved in {save_path_plots} as best_height.png')


            st.write(str(round(dist_height[a_fd],2)))
            st.write(a_fd)
            with st.spinner(f"Finding best-fit cone angle in planes:",show_time=True):
                
                points_red = [[0, 0, 0],
                        [C.x/u.m, C.y/u.m, C.z/u.m],
                            [0, 0, 0+5e8]]

                points_green = [[C.x/u.m-5e8, C.y/u.m, C.z/u.m],
                        [C.x/u.m, C.y/u.m, C.z/u.m],
                        [C.x/u.m, C.y/u.m+6e8, C.z/u.m]]

                [x_red,n_red,z_red,normal] = direcd.get_plane('red',points_red)
                [x_green,n_green,z_green,normal_new] = direcd.get_plane('green',points_green)

                P_x_iter  = P_x[a_fd]
                P_y_iter  = P_y[a_fd]
                P_z_iter  = P_z[a_fd]



                xp,yp,zp = direcd.proj_plane(P_x_iter, P_y_iter,P_z_iter,C.x.to_value(),C.y.to_value(),C.z.to_value(),normal)
                C_plane_1x = np.array([C.x/u.m,xp])
                C_plane_1y = np.array([C.y/u.m,yp])
                C_plane_1z = np.array([C.z/u.m,zp])

                ## Projection of CP to green plane
                xp,yp,zp = direcd.proj_plane(P_x_iter, P_y_iter,P_z_iter,C.x.to_value(),C.y.to_value(),C.z.to_value(),normal_new)
                C_plane_2x = np.array([C.x/u.m,xp])
                C_plane_2y = np.array([C.y/u.m,yp])
                C_plane_2z = np.array([C.z/u.m,zp])


                ## Extension of OC (in red plane)

                x2 = np.array([0, C.x/u.m])
                y2 = np.array([0, C.y/u.m])
                z2 = np.array([0, C.z/u.m])

                length = np.sqrt((x2[0]-x2[-1])**2 + (y2[0]-y2[-1])**2 + (z2[0]-z2[-1])**2)
                unitSlopeX = (x2[0]-x2[-1]) / length
                unitSlopeY = (y2[0]-y2[-1]) / length
                unitSlopeZ = (z2[0]-z2[-1]) / length

                x2_2 = x2[-1] - unitSlopeX * 10e8
                y2_2 = y2[-1] - unitSlopeY * 10e8
                z2_2 = z2[-1] - unitSlopeZ * 10e8

                x2_extended = np.array([x2[-1], x2_2])
                y2_extended = np.array([y2[-1], y2_2])
                z2_extended = np.array([z2[-1], z2_2])

                ## Projection of OC to Green plane

                xp_1,yp_1,zp_1 = direcd.proj_plane(x2_extended[-1], y2_extended[-1],z2_extended[-1],C.x.to_value(),C.y.to_value(),C.z.to_value(),normal_new)
                
                C_plane_3x = np.array([C.x/u.m,xp_1])
                C_plane_3y = np.array([C.y/u.m,yp_1])
                C_plane_3z = np.array([C.z/u.m,zp_1])

                ## Plot OC

                x2 = np.array([0, C.x/u.m])
                y2 = np.array([0, C.y/u.m])
                z2 = np.array([0, C.z/u.m])

                ## Plot CP

                CP1_x = np.array([C.x/u.m, P_x_iter])
                CP1_y = np.array([C.y/u.m, P_y_iter])
                CP1_z = np.array([C.z/u.m, P_z_iter])



                a = np.array([float(C_plane_3x[-1]), float(C_plane_3y[-1]),float(C_plane_3z[-1])]) ## Point C''
                b = np.array([C.x/u.m, C.y/u.m, C.z/u.m])
                c1 = np.array([float(C_plane_2x[-1]), float(C_plane_2y[-1]),float(C_plane_2z[-1])]) ## Point P''

                ba = a - b
                bc = c1 - b
                vec1 = ba
                vec2=bc

                # Print the angle in degrees (positive or negative)

                angle_green = np.round(direcd.angle_pos_neg(vec1,vec2),2)

                normal = [0, 0, 1]
                print(np.cross(ba,bc).dot(normal))

                if (np.cross(ba,bc).dot(normal))<0:
                    angle_green = -angle_green


                a = np.array([x2_extended[-1], y2_extended[-1], z2_extended[-1]]) ## Point C extended
                b = np.array([C.x/u.m, C.y/u.m, C.z/u.m]) ## Point C
                c1 = np.array([float(C_plane_1x[1]), float(C_plane_1y[1]),float(C_plane_1z[1])]) ## Point P

                ba = a-b
                bc = c1-b

                vec1 = ba
                vec2=bc


                angle_red = np.round(direcd.angle_pos_neg(vec1,vec2),2)

                normal = [0, -1, 0]
                print(np.cross(ba,bc).dot(normal))

            

                if (np.cross(ba,bc).dot(normal))<0:
                    angle_red = -angle_red
                

                dim_pix=[]
                for y in range(len(contours)):
                    for k in range(len(contours[y])):
                        i = contours[y][k][0]
                        j = contours[y][k][1]
                        dim_pix.append(smap_mask.pixel_to_world(j * u.pix, i * u.pix))
                
                dim_3d_x=[]
                dim_3d_y=[]
                dim_3d_z=[]
                for i in range(len(dim_pix)):

                    dim_3d_x.append(dim_pix[i].transform_to(frames.HeliographicStonyhurst).cartesian.x.to_value())
                    dim_3d_y.append(dim_pix[i].transform_to(frames.HeliographicStonyhurst).cartesian.y.to_value())
                    dim_3d_z.append(dim_pix[i].transform_to(frames.HeliographicStonyhurst).cartesian.z.to_value())

                
                fig = plt.figure(figsize=(12, 12),constrained_layout=True)
                ax = fig.add_subplot(1, 1, 1, projection='3d',computed_zorder=False)
                x_sphere,y_sphere,z_sphere = direcd.create_sphere(cx=0,cy=0,cz=0,r = Rsun)
                
                CP_x=[]
                CP_y=[]
                CP_z=[]

                for i in range(len(P_x)):

                    CP_x.append(np.array([C.x/u.m, P_x[i]]))
                    CP_y.append(np.array([C.y/u.m, P_y[i]]))
                    CP_z.append(np.array([C.z/u.m, P_z[i]]))

            
                ax.plot_surface(x_sphere, y_sphere, z_sphere, rstride=3, cstride=3,shade=False,alpha=1,rasterized=True,antialiased=True, color = 'yellow',edgecolor='yellow',linewidth=0.5)
                ax.plot_surface(X3_array[a_fd], Y3_array[a_fd], Z3_array[a_fd], rstride=1, cstride=1,alpha=0.5,color='darkgreen')
                
                ax.scatter3D(A_cart.xyz.value[0], A_cart.xyz.value[1], A_cart.xyz.value[2],s=50, c='black', marker='o',zorder=3)
                ax.scatter3D(B_cart.xyz.value[0], B_cart.xyz.value[1], B_cart.xyz.value[2],s=50, c='black', marker='o',zorder=3)
                ax.scatter3D(dim_3d_x, dim_3d_y, dim_3d_z,s=4, c='m', marker='o',zorder=2)
    #ax.text(f_x,f_y-0.05e9,f_z-0.05e9,'C',fontsize=25)
                ax.text(CP_x[a_fd][-1],CP_y[a_fd][-1],CP_z[a_fd][-1],'P',fontsize=25)
                ax.text(B_cart.xyz.value[0], B_cart.xyz.value[1], B_cart.xyz.value[2]-0.05e9,'B',fontsize=25)
                ax.text(A_cart.xyz.value[0], A_cart.xyz.value[1], A_cart.xyz.value[2]+0.03e9,'A',fontsize=25)
                ax.plot3D(CP_x[a_fd],CP_y[a_fd],CP_z[a_fd],linewidth=2,color='black')
                

                for i in range(len(c[10])):
                    ax.scatter3D(c[a_fd].cartesian.x[i].to_value()*10**3, c[a_fd].cartesian.y[i].to_value()*10**3, c[a_fd].cartesian.z[i].to_value()*10**3,s=8, c='g', marker='o')
                    




                Q_2 = SkyCoord(CartesianRepresentation(C.x,C.y,C.z),frame=frames.HeliographicStonyhurst,
                    obstime='7-Jun-2021 04:52:00.620', observer=smap.coordinate_frame.observer)
                Q_2 = Q_2.transform_to(frames.HeliographicStonyhurst)
    #Q = SkyCoord(Lonn*u.deg,Latt*u.deg,Rsun*u.m,observer='earth',obstime=smap.meta['date-obs'],frame=frames.HeliographicStonyhurst)
    #Q = Q.transform_to(frames.HeliographicCarrington)

                north_2=Q_2
                tt, kk = smap_mask.world_to_pixel(Q_2)

                ll = smap_mask.pixel_to_world(tt, kk)

                #all_map = SkyCoord(ll,frame=frames.Helioprojective,obstime=Q.obstime)

                new_frame_int = NorthOffsetFrame(north=north_2)
    #h = all_map.transform_to(new_frame)
                
                for ii in range(-180,180,30):
                    new_frame_coord=[]
                    new_frame_2=[]
                    q=[]
                    hpc_out = sunpy.coordinates.HeliographicStonyhurst()
                    count=0

                    X_cart = []
                    Y_cart = []
                    Z_cart = []
                    X_cart_array=[]
                    Y_cart_array=[]
                    Z_cart_array=[]

                    for iii in range(-90,91,1):
                        new_frame_coord.append(SkyCoord((ii+15)*u.deg, (iii)*u.deg, frame=new_frame_int))
                        new_frame_2.append(new_frame_coord[count].transform_to(hpc_out))
                    #q.append(new_frame_coord[count])
                        q.append(new_frame_2[count].make_3d())
                    #q.append(new_frame_2[count])
                        X_cart.append(q[count].cartesian.x.value*10**3)
                        Y_cart.append(q[count].cartesian.y.value*10**3)
                        Z_cart.append(q[count].cartesian.z.value*10**3)
                        if X_cart[count]>0:
                            X_cart_array.append(X_cart[count])
                            Y_cart_array.append(Y_cart[count])
                            Z_cart_array.append(Z_cart[count])

                            #ax.scatter(X_cart[count],Y_cart[count],Z_cart[count], s=3, color='blue',zorder=4)

                        count =count+1
                    ax.plot3D(X_cart_array, Y_cart_array, Z_cart_array, 'blue',zorder=2)
                
                logo_dir = os.path.join(current_dir, 'logo', 'DIRECD_logo.jpg')
                im = image.imread(logo_dir)
                imagebox = OffsetImage(im, zoom=0.2)
                imagebox.image.axes = ax
                xy = (0, 0)

                ab = AnnotationBbox(imagebox, xy,
                                    xybox=(0.2, 0.2),
                                    xycoords='axes fraction',
                                    boxcoords="axes fraction",
                                    pad=0.5,
                                    frameon=False
                                    )

                ax.add_artist(ab)

                ax.view_init(azim=0,elev=0)
                ax.set_box_aspect([1,1,1], zoom=1.5)
                direcd.set_axes_equal(ax)
                ax.axis('off')

                st.pyplot(fig)

                if save_all_plots_checkbox:
                        fig.savefig(os.path.join(save_path_plots, '3d_cone'+'.png'),facecolor='w',bbox_inches='tight',dpi=300)
                        st.success(f'Plot saved in {save_path_plots} as 3d_cone.png' )

                st.info(f"""
                **DIRECD Results:**  
                - **Height:** `{str(round(dist_height[a_fd],2))}`  
                - **Inclination angle:** `{str(round(angle_2[a_fd],2))}`  
                - **Cone width:** `{str(round(width[a_fd],2))}`
                """)

                st.write('Writing DIRECD results to file')
                row = {
                'Start of Impulsive phase': [begin_impulsive_phase.strftime("%H:%M:%S")],
                'End of Impulsive phase': [end_impulsive_phase.strftime("%H:%M:%S")],
                'Flare Source': [f"{lat_val} {lat_dir}, {lon_val} {lon_dir}"],
                'Best Height': [round(dist_height[a_fd], 2)],
                'Best Incl.': [round(angle_2[a_fd], 2)],
                'Meridional': [round(angle_red, 2)],
                'Equatorial': [round(angle_green, 2)],
                'P_x': P_x[a_fd],
                'P_y': P_y[a_fd],
                'P_z': P_z[a_fd],
                'alpha': alpha_choose_2
                }

                df = pd.DataFrame(row)  # Create DataFrame with single row

                # Save to text file
                info_path = os.path.join(selected_folder_path, 'Events', safe_event, 'DIRECD_results.txt')
                with open(info_path, 'w') as f:
                    f.write(df.to_string(justify='left', index=False))
                
                row_2 = {
                'Cone height': dist_height,
                'Inclination Angle': angle_2,
                'Width': width,
                'P_x': P_x,
                'P_y': P_y,
                'P_z': P_z
                }

                df = pd.DataFrame(row_2) 
                info_path = os.path.join(selected_folder_path, 'Events', safe_event, 'cone_ensemble_data.txt')
                with open(info_path, 'w') as f:
                    f.write(df.to_string(justify='left', index=False))
                    
                


            all_variables = {
            # Inputs (if defined earlier)
            'range_count': range_count,  # Assuming this exists
            'mid_x': mid_x,
            'mid_y': mid_y,
            'mid_z': mid_z,
            'width': width,
            'C': C,  # Astropy SkyCoord object
            
            'Rsun': Rsun,
            'smap': smap,  # SunPy map object (if applicable)


            # Outputs
            'angle_2': angle_2,
            'cone_x': cone_x,
            'cone_y': cone_y,
            'cone_z': cone_z,
            'P_x': P_x,
            'P_y': P_y,
            'P_z': P_z,
            'cone_lat': cone_lat,
            'cone_lon': cone_lon,
            'X3_array': X3_array,
            'Y3_array': Y3_array,
            'Z3_array': Z3_array,
            'height_cone': height_cone,
            'c': c,  # List of SkyCoord objects
            'xe': xe,
            'ye': ye,
            'ze': ze,
            'dist_height': dist_height,
            'h_fd': h_fd,
            'a_fd':a_fd,
            'j': j,  # Loop counter
            }

        # --- Save to a single pickle file ---
            
            output_path = os.path.join(variables_folder,'all_variables.pickle')
            st.write(f'Writing Python Vars : {output_path}')
            with open(output_path, 'wb') as file:
                pickle.dump(all_variables, file)

            st.write(f"All variables saved to {output_path}")

            st.write('completed')
        except Exception as e: 
                st.write(e)
