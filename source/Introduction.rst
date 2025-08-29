Introduction
====================

DIRECD (Dimming Inferred Estimation of CME Direction) is a novel method
to characterize the early CME propagation direction from the expansion of coronal dimmings. 

For more details refer to our article: 

Graphical User Interface
------------

The DIRECD-soft GUI is developed to provide the necessary tools to perform early CME direction analysis using 
coronal dimmings. This package has been built in Python (3.12) with an extensive use of libraries available within
the Python language ecosystem. The GUI of this application has been built based on Streamlit, an open-source Python 
library that provides an easy way to create web applications and tested on windows-based operating system.

Home page
-------------

The application's Home page features a brief introduction to the software alongside key publications detailing the DIRECD method. 
A navigation panel on the left side lists all available pages, allowing users to initiate an analysis by selecting the Dimming Detection
module.

Dimming detection page
--------------

Detecting coronal dimmings is a process that benefits greatly from visual feedback and iterative adjustment. 
The user aims to accurately identify and segment dimming regions by configuring a set of detection parameters and evaluating 
the results against solar observations. DIRECD-soft provides a streamlined solution for this analysis. 
Its intuitive graphical user interface, also built on Streamlit, facilitates an efficient and user-guided detection process.  

The DIRECD-soft web application features a dedicated Dimming Detection page, which is organized into two primary vertical panels. 
The left panel serves as a control center, hosting all input widgets that allow you to configure the detection parameters and 
control the analysis. Conversely, the right panel is dedicated to data visualization, displaying all graphical outputs including 
plots and results to provide immediate visual feedback. This clear separation of controls and outputs enhances the application's 
usability, enabling you to refine your analysis with ease.  

To begin, the user initiates the process by selecting a pre-defined event from the “Event” dropdown menu or by defining a custom event. 
The software currently supports EUV images from SDO/AIA. Key parameters include the event’s date and approximate time, the 
detection time range (which defaults to 180 minutes), the wavelength (193 or 211 Å), and the image cadence (12 to 60 seconds). 
The flare source location is specified in heliographic latitude and longitude and must be within ±60° to be compatible with the 
DIRECD method. Finally, the user selects the threshold for dimming detection from three predefined options (-0.11, -0.15, -0.19) to execute the detection. 
Additional options allow for the automatic saving of plots and overwriting previous results.

.. figure:: images_docs/dimming_detection_page.png
    :align: center
    :scale: 30 %
    :alt: map to buried treasure

   Fig 1. Selection panels of the Dimming Detection page


DIRECD analysis page
--------------

Comparison
---------------