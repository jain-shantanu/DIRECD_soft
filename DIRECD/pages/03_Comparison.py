import streamlit as st
import os
import sunpy
from functions import map_calibration as mp
from functions import direcd_functions as direcd
import tempfile
import subprocess
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import astropy.units as u
from datetime import datetime, timedelta
import shutil
from pathlib import Path
import io
import tkinter as tk
from tkinter import filedialog
import hvpy
from sunpy.util.config import get_and_create_download_dir
from sunpy.time import TimeRange, parse_time
from scipy.ndimage import rotate
import pandas as pd

current_dir = Path(__file__).parent.absolute().parent
logo_folder_short = os.path.join(current_dir, 'logo', 'direcd_short.png')


st.set_page_config(page_title="Comparison", page_icon=logo_folder_short)
st.header("DIRECD: Dimming Inferred Estimation of CME Direction")

st.markdown('---')

def load_fits_data(uploaded_file):
    """Load FITS file and return data and header."""
    with fits.open(uploaded_file) as hdul:
        try:
            data = hdul[0].data
            header = hdul[0].header
        except:
            data = hdul[1].data
            header = hdul[1].header
        return data, header
    

if 'wavelength_val' not in st.session_state:
       
        st.session_state.wavelength_val = 211
if 'cadence_val' not in st.session_state:
    st.session_state.cadence_val = 60

if 'file_calibration' not in st.session_state:
    st.session_state.file_calibration = 'Already have calibrated data'
if 'cone_height' not in st.session_state:
    st.session_state.cone_height = 1

if 'submitted_3' not in st.session_state:
    st.session_state.submitted_3 = False


if 'cor2_map' not in st.session_state:
    st.session_state.cor2_map = None
if 'plotting_done' not in st.session_state:
    st.session_state.plotting_done = False
if 'folder_path' not in st.session_state:
        st.session_state.folder_path = False
if 'sc' not in st.session_state:
        st.session_state.sc = 'LASCO-C2'
if 'obstime' not in st.session_state:
        st.session_state.obstime = None



if st.session_state.folder_path is False:
    def select_folder():
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(master=root)
        root.destroy()
        return folder_path

    selected_folder_path = st.session_state.get("folder_path", None)
    folder_select_button = st.button("Please select a folder where DIRECD results are saved")
    if folder_select_button:
        selected_folder_path = select_folder()
    if selected_folder_path:
        st.write('Selected folder path:', selected_folder_path)
    st.session_state.folder_path = selected_folder_path



# Initialize session state

def detect_parameter_changes(wavelength, cadence,file_calibration, cone_height,sc):
    # Initialize change tracking
    changes = {
        'any_changed': False,
        'wavelength': False,
        'cadence': False,
        'file_calibration': False,
        'cone_height': False,
        'sc': False,
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
    
    
    
    if st.session_state.file_calibration != calibrated_data:
        st.session_state.file_calibration = calibrated_data
        changes['file_calibration'] = True
        changes['changed_params'].append('file_calibration')
    if st.session_state.cone_height != cone_height:
        st.session_state.cone_height= cone_height
        st.session_state.plotting_done = False
        changes['cone_height'] = True  # Changed from 'cone_height_slider'
        changes['changed_params'].append('cone_height')  # Changed from 'cone_height_slider'
    
    if st.session_state.sc != sc:
        st.session_state.sc = sc
        st.session_state.plotting_done = False
        changes['sc'] = True
        changes['changed_params'].append('sc')
    
    # Set overall change flag
    changes['any_changed'] = len(changes['changed_params']) > 0
    
    return changes

   



default_date = "today"
default_time = '20:44'


if st.session_state.folder_path is not False:
    selected_folder_path = st.session_state.folder_path

    try:
        data_event = pd.read_csv(st.session_state.folder_path + 'list_events.csv')
        string_options = (data_event['Event'] + 'T' + data_event['Time']).str.replace('-', '/').tolist()
        string_option_1 = 'New Event'
        string_options.insert(0, string_option_1)
    except:
        string_options='New Event'

    list_events =st.sidebar.selectbox('Event', options = (string_options), index=0,
        placeholder="list_events")
    st.session_state.list_events = list_events

    if st.session_state.list_events is not 'New Event':
        default_date = datetime.strptime(st.session_state.list_events, '%d/%m/%YT%H:%M').date()
        default_time = datetime.strptime(st.session_state.list_events, '%d/%m/%YT%H:%M').time()




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
    
    sc = form.selectbox("Choose spacecraft",('LASCO-C2','STEREO-A'))

    help_txt = '''
    # Is calibrated data available?
    If :green[yes] , select 'Already have calibrated data'.  
    If :red[no], select 'No calibrated data'. The LASCO calibration routine path will be displayed on the screen

    '''

    calibrated_data = form.selectbox("Calibration of files",('No calibrated data', 'Already have calibrated data'), help=help_txt)
    
    submit = form.form_submit_button("Submit")

    help_txt = '''
    # Select cone height (Rsun)
    We extrapolate the cone to LASCO heights with the same width and angle as the best-fit cone. The :red[width] and :red[inclination] are the same as the best-fit cone. 
    '''

    cone_height = st.sidebar.slider("Cone Height", 1.0, 10.0,step=0.1,key="myslider",format="%0.1f",help=help_txt)

    event_dt = mp.create_event_datetime(date_event, time_event)

    if submit:
        
        st.session_state.submitted_3 = True




    if st.session_state.submitted_3:
        changes = detect_parameter_changes(wavelength, cadence,calibrated_data, cone_height,sc)
        current_dir = Path(__file__).parent.absolute().parent
        event_dt = mp.create_event_datetime(date_event, time_event)
        safe_event = event_dt.strftime("%Y-%m-%dT%H:%M:%S").replace(":", "-")
        main_event_folder = os.path.join(selected_folder_path, 'Events', safe_event)
        variables_folder = os.path.join(main_event_folder, 'variables',str(wavelength),str(cadence))
        output_path = os.path.join(variables_folder, 'all_variables.pickle')

        try:
            with open(output_path, 'rb') as file:
                loaded_data = pickle.load(file)
            
        
            range_count = loaded_data.get('range_count', None)  # Use .get() to avoid KeyError if not saved
            
            mid_x = loaded_data.get('mid_x', None)
            mid_y = loaded_data.get('mid_y', None)
            mid_z = loaded_data.get('mid_z', None)
            width = loaded_data.get('width', None)
            C = loaded_data.get('C', None)  # Astropy SkyCoord object
            Rsun = loaded_data.get('Rsun', None)
            smap = loaded_data.get('smap', None)  # SunPy map object (if saved)
            event = loaded_data.get('event', None)


            angle_2 = loaded_data.get('angle_2',None)
            cone_x = loaded_data.get('cone_x',None)
            cone_y = loaded_data.get('cone_y',None)
            cone_z = loaded_data.get('cone_z',None)
            P_x = loaded_data.get('P_x',None)
            P_y = loaded_data.get('P_y',None)
            P_z = loaded_data.get('P_z',None)
            cone_lat = loaded_data.get('cone_lat',None)
            cone_lon = loaded_data.get('cone_lon',None)
            X3_array = loaded_data.get('X3_array',None)
            Y3_array = loaded_data.get('Y3_array',None)
            Z3_array = loaded_data.get('Z3_array',None)
            height_cone = loaded_data.get('height_cone',None)
            c = loaded_data.get('c',None)
            xe = loaded_data.get('xe',None)
            ye = loaded_data.get('ye',None)
            ze = loaded_data.get('ze',None)
            dist_height = loaded_data.get('dist_height',None)
            a_fd = loaded_data.get('a_fd', None)
            h_fd = loaded_data.get('h_fd', None)
            st.session_state.loaded_data = loaded_data
            st.write('DIRECD Data Loaded')
            

        except:
            st.write('No DIRECD data found. Please rerun DIRECD for this event')
            st.stop()

        if changes['any_changed']:
            st.write(f"Parameters changed: {', '.join(changes['changed_params'])}")
            if changes.get('cone_height', False):  # Changed from changes['cone_height']
                cone_plotting = True
            if changes.get('sc', False):  # Changed from changes['cone_height']
                cone_plotting = True
        
        if st.session_state.sc == 'LASCO-C2':     
            if st.session_state.file_calibration == 'No calibrated data':
                obstime = st.text_input("Enter LASCO time (YYYY/MM/DD HH:mm)")
                # event_path = os.path.join(selected_folder_path, 'Events', safe_event, 'lasco.pro')
                # idl_script_path = os.path.join(current_dir, 'IDL', 'lasco.pro')
                # shutil.copy(idl_script_path, event_path)
                # st.write(f'Calibration Routines are provided in {event_path}. Please run it in solarsoft to get calibrated files')
                st.session_state.uploaded_file = None
                st.session_state.cor2_map = None
            else:
                uploaded_file = st.file_uploader("Upload Calibrated LASCO image (LASCO/C2) ", type=['fits', 'fts'],key='uploaded_file')
                obstime= None
                st.session_state.cor2_map = None

            
        else:
            if st.session_state.file_calibration == 'No calibrated data':
                obstime = st.text_input("Enter STEREO time (YYYY/MM/DD HH:mm)")
                
                st.session_state.uploaded_file=None
                st.session_state.cor2_map = None
            else:
                uploaded_file = st.file_uploader("Upload Calibrated STEREO-A image (STEREO-A/COR2) ", type=['fits', 'fts'],key='uploaded_file')
                obstime= None
                st.session_state.cor2_map = None
            
        if obstime is not None:
            if obstime:
                if st.session_state.sc == 'LASCO-C2':
                    cor2_file = hvpy.save_file(hvpy.getJP2Image(parse_time(obstime).datetime,
                                                hvpy.DataSource.LASCO_C2.value),
                                get_and_create_download_dir() + "/COR2.jp2",overwrite=True)
                else:
                    cor2_file = hvpy.save_file(hvpy.getJP2Image(parse_time(obstime).datetime,
                                                hvpy.DataSource.COR2_A.value),
                                get_and_create_download_dir() + "/COR2.jp2",overwrite=True)


                cor2_map = sunpy.map.Map(cor2_file)
                st.session_state.uploaded_file = None
                st.session_state.cor2_map = cor2_map
                st.session_state.plotting_done = False
                
            


        if st.session_state.uploaded_file and st.session_state.uploaded_file is not None:
            
            
            data, header = load_fits_data(st.session_state.uploaded_file)
            try:
                header['date_obs'] = header['date_obs'].strip()
                header['date-obs'] = header['date-obs'].strip()
            except:
                pass

            finally:
                cor2_map = sunpy.map.Map(data, header)
                st.session_state.cor2_map = cor2_map
                rad, R_km, cenX, cenY, nanleft, dist_top = mp.map_info_2(cor2_map)
                st.success('File Uploaded!')
                st.session_state.plotting_done = False


        if st.session_state.uploaded_file and st.session_state.uploaded_file is not None:
            if st.session_state.cor2_map and st.session_state.cor2_map is not None and not st.session_state.plotting_done:
                with st.spinner('Please wait...',show_time=True):
                    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
                    ax = fig.add_subplot(projection=st.session_state.cor2_map)
                    
                    # Get min/max for scaling
                    vmin = np.nanpercentile(data, 1)
                    vmax = np.nanpercentile(data, 99)
                    
                    vavg = (abs(vmin) + abs(vmax))/2
                    img = ax.imshow(data, cmap='gray', origin='lower', vmin=-vavg, vmax=vavg)
                    fig.colorbar(img, ax=ax, label='Intensity')
                    
                    

                    if not st.session_state.plotting_done:
                        try:

                            points_array = round(st.session_state.cone_height*500)
                            
                            CP_3D_x = np.array([C.x.to_value(),P_x[a_fd]])
                            CP_3D_y = np.array([C.y.to_value(),P_y[a_fd]])
                            CP_3D_z = np.array([C.z.to_value(),P_z[a_fd]])

                            length = np.sqrt((CP_3D_x[0]-CP_3D_x[-1])**2 + (CP_3D_y[0]-CP_3D_y[-1])**2 + (CP_3D_z[0]-CP_3D_z[-1])**2)
                            unitSlopeX = (CP_3D_x[0]-CP_3D_x[-1]) / length
                            unitSlopeY = (CP_3D_y[0]-CP_3D_y[-1]) / length
                            unitSlopeZ = (CP_3D_z[0]-CP_3D_z[-1]) / length
                            

                            x2_2 = CP_3D_x[-1] - unitSlopeX * cone_height*Rsun
                            y2_2 = CP_3D_y[-1] - unitSlopeY * cone_height*Rsun
                            z2_2 = CP_3D_z[-1] - unitSlopeZ * cone_height*Rsun
                            

                            CP_3D_x_2 = np.linspace(C.x.to_value(),x2_2,points_array)
                            CP_3D_y_2 = np.linspace(C.y.to_value(),y2_2,points_array)
                            CP_3D_z_2 = np.linspace(C.z.to_value(),z2_2,points_array)


                            cen_line_x = CP_3D_x_2
                            cen_line_y = CP_3D_y_2
                            cen_line_z = CP_3D_z_2
                            


                            plot_cen_line_x=[]
                            plot_cen_line_y=[]
                            plot_cen_line_z=[]
                            for i in range(len(P_x)):

                                plot_cen_line_x.append(np.array([C.x.to_value(), P_x[i]]))
                                plot_cen_line_y.append(np.array([C.y.to_value(), P_y[i]]))
                                plot_cen_line_z.append(np.array([C.z.to_value(), P_z[i]]))
                            P_x_ch = plot_cen_line_x[a_fd]
                            P_y_ch = plot_cen_line_y[a_fd]
                            P_z_ch = plot_cen_line_z[a_fd]


                            length = np.sqrt((P_x_ch[0]-P_x_ch[-1])**2 + (P_y_ch[0]-P_y_ch[-1])**2 + (P_z_ch[0]-P_z_ch[-1])**2)
                            unitSlopeX = (P_x_ch[0]-P_x_ch[-1]) / length
                            unitSlopeY = (P_y_ch[0]-P_y_ch[-1]) / length
                            unitSlopeZ = (P_z_ch[0]-P_z_ch[-1]) / length

                            
                            
                            x2_2 = P_x_ch[-1] - unitSlopeX * cone_height*Rsun
                            y2_2 = P_y_ch[-1] - unitSlopeY * cone_height*Rsun
                            z2_2 = P_z_ch[-1] - unitSlopeZ * cone_height*Rsun


                            P_x_line = np.linspace(P_x_ch[0],x2_2,50)
                            P_y_line = np.linspace(P_y_ch[0],y2_2,50)
                            P_z_line= np.linspace(P_z_ch[0],z2_2,50)

                            counter = 0
                            new_cen_x_new=[]
                            new_cen_y_new=[]
                            new_cen_z_new=[]

                            f_x = C.x.to_value()
                            f_y = C.y.to_value()
                            f_z = C.z.to_value()
                            

                            #3.60
                            #6.94
                            for i in range(len(cen_line_x)):
                                
                                if np.round((np.sqrt((cen_line_x[i]-f_x)**2 + (cen_line_y[i]-f_y)**2 + (cen_line_z[i]-f_z)**2)/Rsun),2)== cone_height:
                                    
                                    new_cen_x_new.append(cen_line_x[i])
                                    new_cen_y_new.append(cen_line_y[i])
                                    new_cen_z_new.append(cen_line_z[i])
                                    counter = counter+1
                            





                        
                            #angle=[]
                            angle_2_new=[]
                            #angle_3=[]

                            cone_x_new=[]
                            cone_y_new=[]
                            cone_z_new=[]

                            P_x_new=[]
                            P_y_new=[]
                            P_z_new=[]

                            cone_lat_new=[]
                            cone_lon_new=[]

                            X3_array_new=[]
                            Y3_array_new=[]
                            Z3_array_new=[]
                            height_new=[]

                        

                            j = 0;
                            for iter in range(1):

                            #for iter in range(0,5):


                                h = ((np.sqrt((new_cen_x_new[0] - f_x)**2 + (new_cen_y_new[0] - f_y)**2 + (new_cen_z_new[0] - f_z)**2)))
                                
                                height_new.append(h)

                            #     Bubble_x = (OA_3D_x_2[index_height[iter]]+OB_3D_x_2[index_height[iter]])/2
                            #     Bubble_y = (OA_3D_y_2[index_height[iter]]+OB_3D_y_2[index_height[iter]])/2
                            #     Bubble_z = (OA_3D_z_2[index_height[iter]]+OB_3D_z_2[index_height[iter]])/2
                                
                                Bubble_x = new_cen_x_new[0]
                                Bubble_y = new_cen_y_new[0]
                                Bubble_z = new_cen_z_new[0]


                                P_x_new.append(Bubble_x)
                                P_y_new.append(Bubble_y)
                                P_z_new.append(Bubble_z)

                                l1 =Bubble_x-f_x
                                n1 = Bubble_y -f_y
                                m1 = Bubble_z -f_z



                                l2 = -(0-f_x)
                                n2 = -(0-f_y)
                                m2 = -(0-f_z)

                                vec1=[(l2),(m2),(n2)]


                                vec2=[(l1),(m1),(n1)]

                                angle_2_new.append(np.degrees(np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))))
                                theta = width[a_fd]/2
                                
                                #theta = 89.87/2
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
                                xx = (f_x - P_x_new[j])
                                yy = (f_y- P_y_new[j])
                                zz = (f_z - P_z_new[j])

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
                                X3_new = (f_x - (X1*np.cos(phix)-Y1*np.sin(phix)));
                                Y3_new = (f_y - (X1*np.sin(phix)+Y1*np.cos(phix)));
                                Z3_new = (f_z -  (Z1));
                        
                            cone_coords = [
                                direcd.km_to_pixel_coords(np.array([0, Y3_new[i][j], Z3_new[i][j]]), rad, R_km*10**3, cenX, cenY, cor2_map.data.shape)
                            
                                for i in range(len(X3_new)) 
                                for j in range(len(X3_new[i]))
                                ]

                            line_fit_coords = [
                                direcd.km_to_pixel_coords(np.array([0, P_y_line[i], P_z_line[i]]), rad, R_km*10**3, cenX, cenY, cor2_map.data.shape)
                                for i in range(len(P_x_line))
                                ]
                            

                            cone_x = [c[0] for c in cone_coords] * u.pix
                            cone_y = [c[1] for c in cone_coords] * u.pix

                            line_x = [c[0] for c in line_fit_coords] * u.pix
                            line_y = [c[1] for c in line_fit_coords] * u.pix

                            # Plot all points at once
                            ax.plot_coord(cone_x, cone_y, 'o', color='blue', markersize=1)
                            ax.plot_coord(line_x, line_y, 'o', color='magenta', markersize=1)
                            ax.grid(False)
                            
                            # for i in range(len(cone_coords)):
                            #     ax.plot_coord(cone_coords[i][0]*u.pix, cone_coords[i][1]*u.pix, 'o', color='blue', markersize=1)
                            #     ax.grid(False)    
                        
                            # for i in range(len(line_fit_coords)):
                            #     ax.plot_coord(line_fit_coords[i][0]*u.pix, line_fit_coords[i][1]*u.pix, 'o', color='magenta', markersize=1)

                            ax.set_xlabel('X (Arcsec)')
                            ax.set_ylabel('Y (Arcsec)')
                            ax.set_title('FITS Image')

                            img = io.BytesIO()
                            fn = 'DIRECD on Coronagraph.png'
                            plt.savefig(img, format='png',bbox_inches='tight', dpi=300)

                            btn = st.download_button(
                            label="Download image",
                            data=img,
                            file_name=fn,
                            mime="image/png"
                            )
                



                                
                            st.session_state.plotting_done = True
                                # plt.close(fig)
                                # st.pyplot(fig)
                                





                        except:
                                st.write('Error generating cones')
                                st.session_state.plotting_done = False
                                
                    plt.close(fig)
                    st.pyplot(fig)
        
        else:
            if st.session_state.cor2_map and st.session_state.cor2_map is not None and not st.session_state.plotting_done:
                if st.session_state.sc=='LASCO-C2':
                    if abs(cor2_map.meta.get('crota1', 0)) <10:
                        rotated_data = rotate(cor2_map.data, angle=cor2_map.meta.get('crota1', 0), reshape=False)
                    else: 
                        rotated_data = rotate(cor2_map.data, angle=180 - cor2_map.meta.get('crota1', 0), reshape=False)
                    st.write(cor2_map.meta.get('crota1', 0))

# Create new metadata with updated rotation
                    new_meta = cor2_map.meta.copy()
                    new_meta['crota1'] = (cor2_map.meta.get('crota1', 0) - cor2_map.meta.get('crota1', 0))
                    new_meta['crota2'] = (cor2_map.meta.get('crota2', 0)  - cor2_map.meta.get('crota2', 0))
                    new_meta['crota'] = (cor2_map.meta.get('crota', 0) - cor2_map.meta.get('crota', 0))

# Create new map with rotated data
                    rotated_map = sunpy.map.Map(rotated_data, new_meta)
                    cor2_map = rotated_map
                    st.write(cor2_map.meta.get('crota1', 0))
                with st.spinner('Please wait...',show_time=True):
                    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
                    ax = fig.add_subplot(projection=st.session_state.cor2_map)
                    data = cor2_map.data
                    header = cor2_map.meta
                    # Get min/max for scaling
                    vmin = np.nanpercentile(data, 1)
                    vmax = np.nanpercentile(data, 97)
                    
                    
                    img = ax.imshow(data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                    fig.colorbar(img, ax=ax, label='Intensity')
                    
                    

                    if not st.session_state.plotting_done:
                        try:

                            points_array = round(cone_height*500)
                            
                            CP_3D_x = np.array([C.x.to_value(),P_x[a_fd]])
                            CP_3D_y = np.array([C.y.to_value(),P_y[a_fd]])
                            CP_3D_z = np.array([C.z.to_value(),P_z[a_fd]])

                            length = np.sqrt((CP_3D_x[0]-CP_3D_x[-1])**2 + (CP_3D_y[0]-CP_3D_y[-1])**2 + (CP_3D_z[0]-CP_3D_z[-1])**2)
                            unitSlopeX = (CP_3D_x[0]-CP_3D_x[-1]) / length
                            unitSlopeY = (CP_3D_y[0]-CP_3D_y[-1]) / length
                            unitSlopeZ = (CP_3D_z[0]-CP_3D_z[-1]) / length
                            

                            x2_2 = CP_3D_x[-1] - unitSlopeX * cone_height*Rsun
                            y2_2 = CP_3D_y[-1] - unitSlopeY * cone_height*Rsun
                            z2_2 = CP_3D_z[-1] - unitSlopeZ * cone_height*Rsun
                            

                            CP_3D_x_2 = np.linspace(C.x.to_value(),x2_2,points_array)
                            CP_3D_y_2 = np.linspace(C.y.to_value(),y2_2,points_array)
                            CP_3D_z_2 = np.linspace(C.z.to_value(),z2_2,points_array)


                            cen_line_x = CP_3D_x_2
                            cen_line_y = CP_3D_y_2
                            cen_line_z = CP_3D_z_2
                            


                            plot_cen_line_x=[]
                            plot_cen_line_y=[]
                            plot_cen_line_z=[]
                            for i in range(len(P_x)):

                                plot_cen_line_x.append(np.array([C.x.to_value(), P_x[i]]))
                                plot_cen_line_y.append(np.array([C.y.to_value(), P_y[i]]))
                                plot_cen_line_z.append(np.array([C.z.to_value(), P_z[i]]))
                            P_x_ch = plot_cen_line_x[a_fd]
                            P_y_ch = plot_cen_line_y[a_fd]
                            P_z_ch = plot_cen_line_z[a_fd]


                            length = np.sqrt((P_x_ch[0]-P_x_ch[-1])**2 + (P_y_ch[0]-P_y_ch[-1])**2 + (P_z_ch[0]-P_z_ch[-1])**2)
                            unitSlopeX = (P_x_ch[0]-P_x_ch[-1]) / length
                            unitSlopeY = (P_y_ch[0]-P_y_ch[-1]) / length
                            unitSlopeZ = (P_z_ch[0]-P_z_ch[-1]) / length

                            
                            
                            x2_2 = P_x_ch[-1] - unitSlopeX * cone_height*Rsun
                            y2_2 = P_y_ch[-1] - unitSlopeY * cone_height*Rsun
                            z2_2 = P_z_ch[-1] - unitSlopeZ * cone_height*Rsun


                            P_x_line = np.linspace(P_x_ch[0],x2_2,50)
                            P_y_line = np.linspace(P_y_ch[0],y2_2,50)
                            P_z_line= np.linspace(P_z_ch[0],z2_2,50)

                            counter = 0
                            new_cen_x_new=[]
                            new_cen_y_new=[]
                            new_cen_z_new=[]

                            f_x = C.x.to_value()
                            f_y = C.y.to_value()
                            f_z = C.z.to_value()
                            

                            #3.60
                            #6.94
                            for i in range(len(cen_line_x)):
                                
                                if np.round((np.sqrt((cen_line_x[i]-f_x)**2 + (cen_line_y[i]-f_y)**2 + (cen_line_z[i]-f_z)**2)/Rsun),2)== cone_height:
                                    
                                    new_cen_x_new.append(cen_line_x[i])
                                    new_cen_y_new.append(cen_line_y[i])
                                    new_cen_z_new.append(cen_line_z[i])
                                    counter = counter+1
                            





                        
                            #angle=[]
                            angle_2_new=[]
                            #angle_3=[]

                            cone_x_new=[]
                            cone_y_new=[]
                            cone_z_new=[]

                            P_x_new=[]
                            P_y_new=[]
                            P_z_new=[]

                            cone_lat_new=[]
                            cone_lon_new=[]

                            X3_array_new=[]
                            Y3_array_new=[]
                            Z3_array_new=[]
                            height_new=[]

                        

                            j = 0;
                            for iter in range(1):

                            #for iter in range(0,5):


                                h = ((np.sqrt((new_cen_x_new[0] - f_x)**2 + (new_cen_y_new[0] - f_y)**2 + (new_cen_z_new[0] - f_z)**2)))
                                
                                height_new.append(h)

                            #     Bubble_x = (OA_3D_x_2[index_height[iter]]+OB_3D_x_2[index_height[iter]])/2
                            #     Bubble_y = (OA_3D_y_2[index_height[iter]]+OB_3D_y_2[index_height[iter]])/2
                            #     Bubble_z = (OA_3D_z_2[index_height[iter]]+OB_3D_z_2[index_height[iter]])/2
                                
                                Bubble_x = new_cen_x_new[0]
                                Bubble_y = new_cen_y_new[0]
                                Bubble_z = new_cen_z_new[0]


                                P_x_new.append(Bubble_x)
                                P_y_new.append(Bubble_y)
                                P_z_new.append(Bubble_z)

                                l1 =Bubble_x-f_x
                                n1 = Bubble_y -f_y
                                m1 = Bubble_z -f_z



                                l2 = -(0-f_x)
                                n2 = -(0-f_y)
                                m2 = -(0-f_z)

                                vec1=[(l2),(m2),(n2)]


                                vec2=[(l1),(m1),(n1)]

                                angle_2_new.append(np.degrees(np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))))
                                theta = width[a_fd]/2
                                
                                #theta = 89.87/2
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
                                xx = (f_x - P_x_new[j])
                                yy = (f_y- P_y_new[j])
                                zz = (f_z - P_z_new[j])

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
                                X3_new = (f_x - (X1*np.cos(phix)-Y1*np.sin(phix)));
                                Y3_new = (f_y - (X1*np.sin(phix)+Y1*np.cos(phix)));
                                Z3_new = (f_z -  (Z1));
                            
                            if st.session_state.sc=='LASCO-C2':
                                str_sc = 'SOHO'
                            else:
                                str_sc = 'STEREO-A'

                            cone_coords = [
                                cor2_map.world_to_pixel(
                                direcd.convert_3d_km_to_sc(X3_new[i][j], Y3_new[i][j], Z3_new[i][j], obstime, str_sc))
                                for i in range(len(X3_new)) 
                                for j in range(len(X3_new[i]))
                                ]
                           
                            line_fit_coords = [
                                cor2_map.world_to_pixel(
                                direcd.convert_3d_km_to_sc(P_x_line[i], P_y_line[i], P_z_line[i], obstime, str_sc))
                                for i in range(len(P_x_line))
                                ]
    
                            

                            cone_x = [c[0] for c in cone_coords]*u.pix
                            cone_y = [c[1] for c in cone_coords]*u.pix

                            line_x = [c[0] for c in line_fit_coords]*u.pix
                            line_y = [c[1] for c in line_fit_coords]*u.pix

                            # Plot all points at once
                            ax.plot_coord(cone_x, cone_y, 'o', color='blue', markersize=1)
                            ax.plot_coord(line_x, line_y, 'o', color='magenta', markersize=1)
                            ax.grid(False)
                            
                            # for i in range(len(cone_coords)):
                            #     ax.plot_coord(cone_coords[i][0]*u.pix, cone_coords[i][1]*u.pix, 'o', color='blue', markersize=1)
                            #     ax.grid(False)    
                        
                            # for i in range(len(line_fit_coords)):
                            #     ax.plot_coord(line_fit_coords[i][0]*u.pix, line_fit_coords[i][1]*u.pix, 'o', color='magenta', markersize=1)

                            ax.set_xlabel('X (Arcsec)')
                            ax.set_ylabel('Y (Arcsec)')
                            ax.set_title(f'FITS Image_{obstime} on {str_sc} at {cone_height} Rsun')

                            img = io.BytesIO()
                            fn = 'DIRECD on Coronagraph.png'
                            plt.savefig(img, format='png',bbox_inches='tight', dpi=300)

                            btn = st.download_button(
                            label="Download image",
                            data=img,
                            file_name=fn,
                            mime="image/png"
                            )
                



                                
                            st.session_state.plotting_done = True
                                # plt.close(fig)
                                # st.pyplot(fig)
                                





                        except Exception as e: 
                                st.write(e)
                                st.session_state.plotting_done = False
                                
                    plt.close(fig)
                    st.pyplot(fig)
           
            
        




