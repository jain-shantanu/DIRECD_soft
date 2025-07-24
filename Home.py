import streamlit as st
import os
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
    # st.markdown("""
    #             **Citation**: Please cite the following paper (https://doi.org/10.1051/0004-6361/202347927)
    #             """)
    
    # st.markdown('---')

st.set_page_config(
    page_title="DIRECD - Main Page",
    page_icon=logo_folder  # Direct path to image
)

st.header("DIRECD: Dimming Inferred Estimation of CME Direction")

st.markdown('---')

main_text()

