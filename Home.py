import streamlit as st
import os
from pathlib import Path
current_dir = Path(__file__).parent.absolute()
logo_folder = os.path.join(current_dir, 'logo', 'DIRECD_logo.jpg')

def main_text():
	st.subheader('About this application:')
	st.markdown("""
                   _DIRECD_  is an open-source software package that can be used to
                   reconstruct estimate Coronal Mass Ejections (CMEs) direction from coronal dimming information.
                   The tool implements the dimming detection using region growing algorithm on SDO/AIA images.
                """)
	right, left = st.columns((1, 1))
	left.image(logo_folder)
 
st.set_page_config(
    page_title="DIRECD - Main Page",
    page_icon=logo_folder  # Direct path to image
)

st.header("DIRECD: Dimming Inferred Estimation of CME Direction")

st.markdown('---')

main_text()

