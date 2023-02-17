# Digital Image Analysis for Supersaturation Induced Bubble Nucleation
This repository contains the main source code and experimental data sets reported in[^1] and[^2].

The main source code located in `DIA.py` and helper functions in `DIA_functions.py` are written to detect bubbles nucleating on a porous surface due to super-saturation. 

## How to use

### Installation

Install the packages listed in `requirements.txt`. The code was prepared for use with Python>=3.10.

### Setting up

In `DIA.py`, change the user input to point to the desired folders that contain the experimental data sets.

### Run

Simply run `python3 DIA.py`, the output is found in a subfolder `results` inside the data set folder.

## Generated images

In our work, we validate the method using generated images. The source code as well as various data sets are found in the folder `generated`

## Sample holder

In our work, we refer to a 3D printed sample holder. The CAD files are contained in the folder `sample_holder`.


# References
[^1]: A. Battistella, and I. Roghair and M. van Sint Annaland, _Chem.Eng.Sci._, 2023 (submitted for review)
[^2]: A. Battistella, [Hydrodynamics, Mass Transfer and Phase Transition in Bubbly Flows](https://research.tue.nl/en/publications/hydrodynamics-mass-transfer-and-phase-transition-in-bubbly-flows), PhD Thesis, Eindhoven University of Technology.
