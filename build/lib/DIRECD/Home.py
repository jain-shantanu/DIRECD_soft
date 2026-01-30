import streamlit as st
import os
from pathlib import Path
current_dir = Path(__file__).parent.absolute()
logo_folder = os.path.join(current_dir, 'logo', 'DIRECD_logo.jpg')
logo_folder_short = os.path.join(current_dir, 'logo', 'direcd_short.png')

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
    page_icon=logo_folder_short  # Direct path to image
)

st.header("DIRECD: Dimming Inferred Estimation of CME Direction")

st.markdown('---')

main_text()

st.markdown(f"You are running the kernal on the directory {current_dir}")

url = "https://direcd-soft.readthedocs.io/en/latest/index.html"
st.markdown('Link to the Manual: [link](%s)' % url)

st.markdown("""
                **Citation**: Please cite the following paper for the software (https://doi.org/10.1051/0004-6361/202347927)
                """)
st.markdown("""
**Details**:
- **Introducing DIRECD**: (https://doi.org/10.1051/0004-6361/202347927)
- **Application of DIRECD**: (https://doi.org/10.1051/0004-6361/202452324), (https://doi.org/10.1051/0004-6361/202451777)
- **Statistical Application of DIRECD**: (https://doi.org/10.3847/1538-4365/ae0a27)
""")
    
st.markdown('---')
