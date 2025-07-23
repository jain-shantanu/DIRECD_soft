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



st.set_page_config(page_title="Application of DIRECD")
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
    st.session_state.cone_height = 2

if 'submitted' not in st.session_state:
    st.session_state.submitted = False


if 'cor2_map' not in st.session_state:
    st.session_state.cor2_map = None
if 'plotting_done' not in st.session_state:
    st.session_state.plotting_done = False





# Initialize session state

def detect_parameter_changes(wavelength, cadence,file_calibration, cone_height):
    # Initialize change tracking
    changes = {
        'any_changed': False,
        'wavelength': False,
        'cadence': False,
        'file_calibration': False,
        'cone_height': False,
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
    
    
    
    # Set overall change flag
    changes['any_changed'] = len(changes['changed_params']) > 0
    
    return changes

   



default_date = "today"
default_time = '20:44'
try:
    if st.session_state.list_events == '26/11/2011T06:09':
        default_date = datetime.strptime('26 11 2011', '%d %m %Y')
        default_time = datetime.strptime('06:09', '%H:%M')
except:
    pass



form = st.sidebar.form("Event_selection")
date_col, time_col = form.columns([5, 4])
date_str_min = "01/01/2009"
parsed_date = datetime.strptime(date_str_min, "%d/%m/%Y").date()
with date_col:
    date_event = st.date_input("Date",value=default_date,min_value=parsed_date,key='input_date')  

with time_col:
    time_event = st.time_input("Time",value=default_time,step=60,key='input_time') 


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
   
sc = form.selectbox("Choose spacecraft",('LASCO'))

calibrated_data = form.selectbox("Calibration of files",('Already have calibrated data','No calibrated data'))
   
submit = form.form_submit_button("Submit")

cone_height = st.sidebar.slider("Cone Height", 1, 10,2,key="myslider")

event_dt = mp.create_event_datetime(date_event, time_event)

if submit:
    
    st.session_state.submitted = True




if st.session_state.submitted:
    changes = detect_parameter_changes(wavelength, cadence,calibrated_data, cone_height)
    current_dir = os.getcwd()
    event_dt = mp.create_event_datetime(date_event, time_event)
    safe_event = event_dt.strftime("%Y-%m-%dT%H:%M:%S").replace(":", "-")
    main_event_folder = os.path.join(current_dir, 'Events', safe_event)
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
    
    if st.session_state.file_calibration == 'No calibrated data':
        event_path = os.path.join(current_dir, 'Events', safe_event, 'lasco.pro')
        idl_script_path = os.path.join(current_dir, 'IDL', 'lasco.pro')
        os.replace(idl_script_path, event_path)
        st.write(f'Calibration Routines are provided in {event_path}. Please run it in solarsoft to get calibrated files')
        st.session_state.uploaded_file = None
        st.session_state.cor2_map = None
        st.session_state.plotting_done = False
        
        st.stop()
        
    else:
        uploaded_file = st.file_uploader("Upload Calibrated LASCO image (LASCO) ", type=['fits', 'fts'],key='uploaded_file')
        


    if st.session_state.uploaded_file and st.session_state.uploaded_file is not None:
        
        
        data, header = load_fits_data(st.session_state.uploaded_file)
        cor2_map = sunpy.map.Map(data, header)
        st.session_state.cor2_map = cor2_map
        rad, R_km, cenX, cenY, nanleft, dist_top = mp.map_info_2(cor2_map)
        st.success('File Uploaded!')
        st.session_state.plotting_done = False



    
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

                    points_array = st.session_state.cone_height*500
                    
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



                        
                    st.session_state.plotting_done = True
                        # plt.close(fig)
                        # st.pyplot(fig)
                        





                except:
                        st.write('Error generating cones')
                        st.session_state.plotting_done = False
                        
            plt.close(fig)
            st.pyplot(fig)
        
        # Just show the base image
           
            
        




