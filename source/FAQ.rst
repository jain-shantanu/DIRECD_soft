
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
     
.. code-block:: bash
    pip install --upgrade pip
        
and that you have a working internet connection.


Dimming Detection
------------

**Q: The software can't find or load the solar data. What's wrong?**
    A: Please check the following:

        *File Path: Ensure you have placed the data files in the correct 'fits' subfolder for your event.

        Events/
        └── YYYY-MM-DDTHH-MM-SS/  (event timestamp)
            └── fits/
                └── wavelength/
                    └── cadence/
                    
        *Data Source: Confirm that the data was downloaded from a supported source like JSOC and is in a compatible format (.fits/.fts).


DIRECD Analysis
------------

Comparing with Coronagraphs
------------

Others
------------

**Q: Where can I get help if I encounter a bug or have a question?**

    Please report any bugs or issues you find by opening an issue on our GitHub repository or 
    contacting our team at direcd.soft@gmail.com 
    Be sure to include a description of the problem and any error messages you received.

