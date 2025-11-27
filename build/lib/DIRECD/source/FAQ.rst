
Frequently asked Questions
=================

DIRECD (Dimming Inferred Estimation of CME Direction) is a novel method
to characterize the early CME propagation direction from the expansion of coronal dimmings. 

For more details refer to our article: 


Installation
------------

**Q: What do I need to run DIRECD?**
    Before installing DIRECD, please ensure your system has the following:

    Python: Version 3.10 or higher.

    Pip: The Python package installer, which usually comes with Python.

**Q: How do I install DIRECD?**
    DIRECD installation requires certain packages and dependencies that may interfere with the packages already installed on user computer.
    We recommend setting up a virtual environment using conda and installing DIRECD.

.. code-block:: bash

    # Create the virtual environment
    conda create --name DIRECD python=3.10

    # Activate the environment
    conda activate DIRECD

    # Install the required packages using pip
    pip install DIRECD

**Q: I'm getting errors during installation about missing packages. What should I do?**
     This typically means a dependency failed to install automatically. The most reliable solution is to install DIRECD through pip, as it handles dependencies for you. If problems persist, ensure your pip is up-to-date
     and that you have a working internet connection. 
     
.. code-block:: bash
    pip install --upgrade pip


Dimming Detection
------------

**Q: What are the different parameters to start the dimming detection?**
    A: On the left panel, there are several options that define the dimming detection parameters:

        * Date: Start date of the event to be analyzed

        * Time: Start time of the event to be analyzed. The start time is the flare start time and can be obtained from open 
        sources such as GOES/XRT flare catalog. The base time is automatically chosen as 30 minutes before start time.

        * Time range of Detection: Minimum range for a good detection is 120 minutes from event start, default is 180 mins.

        * Wavelength/Cadence: Wavelength and Cadence of SDO/AIA data (default is 211 A and 1 minute)

        * Flare Source: Flare origin in HEEQ lat/lon coordinates

        * LBR Threshold: Threshold for region-growing dimming detection. Stronger threshold results in stricter 
        dimming detection. (Default = -0.15)


**Q: The software can't find or load the solar data. What's wrong?**
    A: Please check the following:

        * File Path: Ensure you have placed the data files in the correct 'fits' subfolder for your event.

        Events/
            └── YYYY-MM-DDTHH-MM-SS/  (event timestamp)
                └── fits/
                    └── wavelength/
                        └── cadence/

        * Data Source: Confirm that the data was downloaded from a supported source like JSOC and is in a compatible format (.fits/.fts).

**Q: The calibration process produces errors or warnings**
    A: The most common causes of errors/warnings in calibration routine could be:
        1. Incorrect FITS file headers

        2. Missing metadata in downloaded files

        3. Corrupted download files

        To troubleshoot these steps, the users can:

        1. Enable the "Overwrite Raw fits" option to force redownload

        2. Check that all files have consistent metadata

        3. Verify the files are complete SDO/AIA Level 1 data

        In case of a specific error, please reach out to us.

**Q: Dimming Detection produces unexpected results or errors**
    A: Unexpected results and/or errors in dimming detection can occur due to incorrect flare coordinates 
    or insufficient time range of detection. For proper dimming detection, ensure the flare latitude and longitude are correctly specified with proper direction (North/South, East/West)
    and the time range covers at least 120+ minutes after the event.

**Q: The application runs slowly, what to do?** 
    A: The GUI is highly dependent on internet speed. If the application runs slowly, please check your internet connection.
    Consider increasing cadence for faster processing or decreasing the detection time range.


DIRECD Analysis
------------

**Q: What is the purpose of the timing map?**
    A: The timing map shows the progression of dimming regions over time, with the "End of Impulsive Phase" representing the most developed dimming pattern for analysis.

**Q: What do the different cone parameters represent?**
    A: 
    * Height: Estimated CME height in solar radii (Rsun)
    * Width: Angular width of the CME cone in degrees
    * Inclination angle β: Inclination angle of the CME propagation direction

**Q: I get the error: "Edge not found" or script stops during edge detection. What to do in this case?**
    A: The "edge not found" error can happen due to many reasons. If Edge 1 is not found, it usually occurs
    due to insufficient dimming signature or poor detection. Users can try the following troubleshooting methods 
    to see if it helps:

    * Verify the flare source coordinates are accurate
    * Check if the timing map shows clear dimming
    * Try adjusting the "Time to Analyze Map" parameter
    * Use manual edge detection option

    If Edge 1 is found but Edge 2 is not found, then the script automatically takes flare source as the second cone edge
    and continues to the next step. Users also have an option to manually define edges 1 and 2 using the manual edge detection option.

**Technical Notes**

* The application requires SunPy, Astropy, and other astronomical Python packages
* Processing times vary from 5-30 minutes depending on parameters and system capabilities
* Results are saved in the Events directory with timestamps for reproducibility


Comparing with Coronagraphs
------------

**Q: What LASCO data format is required?**
    A: The tool requires calibrated LASCO C2 or C3 FITS files with standard SolarSoft header information.

**Q: I don't have calibrated data, where can I get it?**
    A: At this moment, it's not possible to calibrate LASCO data using astropy/sunpy libraries in python.
    However, we provide IDL/Solarsoft routines with the package where the users can calibrate the LASCO C2 data for DIRECD analysis.
    When the user chooses "I don't have calibrated data", the pro file is saved in:

    Events/
        └── YYYY-MM-DDTHH-MM-SS/  (event timestamp)
                └── lasco.pro
**Q: What does the cone height slider control?**
    A: The cone height slider adjusts the height (in solar radii) of the best-fit cone
    keeping width and inclination angle same, allowing you to see how the CME would appear at different heights in the corona.

**Q: Projected cone position doesn't match LASCO CME position**
    A: This may happen due to improper calibration of LASCO fits files and/or deflection between different heights (since DIRECD calculates CME direction at low heights while coronagraphs show CME at higher heights).
    Other reasons could be bad dimming detection due to presence of other solar features nearby. Such events may not work with automated DIRECD tool and may require
    careful analysis.

**Q: How do I interpret the matching/ non-matching of DIRECD cone with coronagraph?**
    A: Below is a general interpretation guideline, although each event is case-specific and may require more analysis:
    
    * Good agreement: When the projected cone aligns with the actual CME structure in LASCO imagery.
    * Partial agreement: If only parts of the cone match, consider whether:

        - The CME has undergone rotation or deflection
        - The cone model needs adjustment
        - There are multiple CME components, or presence of secondary dimmings.

    * Poor agreement: Significant mismatches may indicate:

        - Incorrect flare source identification
        - Complex CME structure not captured by the simple cone model
        - Data quality or dimming detection issues


Others
------------

**Q: Where can I get help if I encounter a bug or have a question?**

    Please report any bugs or issues you find by opening an issue on our GitHub repository or 
    contacting our team at direcd.soft@gmail.com 
    Be sure to include a description of the problem and any error messages you received.

