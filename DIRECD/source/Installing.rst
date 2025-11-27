.. _installing-direcd:

Installing DIRECD
=================

Requirements
------------

Python 3.10 or above is required to install and run DIRECD.

If you do not have Python installed already, use these `instructions <https://www.python.org/downloads>`_ to download and install it.

Create Virtual Environment
--------------------------

.. tip::

  We recommend to create a virtual environment before installing ``DIRECD``.

If you use `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `Anaconda <https://www.anaconda.com/download>`_
you can run ``DIRECD`` creating a virtual environment first with `Conda <https://docs.conda.io/en/latest/>`_.


**Creating, Activating, and Deactivating a Conda Environment:**

To create a new Conda environment called "PyThea", and then activate it follow these steps:

1. Open a terminal or command prompt.

2. Use the following command to create a new Conda environment named "DIRECD" and install Python:

   .. code-block:: bash

      conda create --name DIRECD python=3.10

   You can replace "3.10" with the desired Python version.

3. Use the following command to activate the "DIRECD" environment:

   .. code-block:: bash

      conda activate DIRECD

   After executing this command, your prompt should change to indicate that the "DIRECD" environment is active.

You can now run ``DIRECD`` and work within this isolated environment when using this software package.

To deactivate the "DIRECD" Conda environment and return to the base environment use,

   .. code-block:: bash

      conda deactivate

 The environment will be deactivated, and you will return to the base Conda environment.

Run DIRECD
----------------

.. code-block:: bash

  # Create the virtual environment
  conda create --name DIRECD python=3.10

  # Activate the environment
  conda activate DIRECD

  # Install the required packages using pip
  pip install DIRECD

.. warning::

  Currently install with ``conda install DIRECD`` is not suppoted.