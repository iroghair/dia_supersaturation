#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is a Digital Image Analysis (DIA) tool relying on opencv's 
Hough Circle Transform in order to detect bubbles nucleating/growing
on a substrate. This still requires a lot of user input in order to
detect bubbles properly and must cycle between three bubble sizes.

Last modification: 08/02/2021
Authors: Alessandro Battistella, Fabio Mirisola
"""

# Imports
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as sio

# DIA functions (must have the corresponding functions.py in the same folder!)
import DIA_functions as DIA

###################
# Here user input #
###################

# File names
base_folder = '../exp' # Folder that contains all experiments in separate folders
data_folder = 'default_pore_size_ss_0.04' # Folder containing files to process in subfolder: data
img_base = 't' # Base name of files to process
background =  'background_0001.tif' # Background image (path to image)
output_folder = 'results' # Where to put the results

# Crop parameters
Crop = True
Xcoord = 400
YCoord = 181
Width = 1370
Height = 1457
angle = 5

# Parameters: (for a list of what they do, refer to each of the 
# functions of opencv/skimage in order to understand)

# parameters large bubbles detection 
noise_large = {'min_size':500, 'connectivity':1000}
holes_large = {'iterations':4}
hough_large = {'param1':1000, 'param2':15 , 'minRadius':35 , 'maxRadius':50}

# parameters medium bubbles detection
noise_medium = {'min_size':250, 'connectivity':1000}
holes_medium = {'iterations':7}
hough_medium = {'param1':375, 'param2':13 , 'minRadius':17 , 'maxRadius':35}

# parameters small bubbles detection
max_radius_small = 25
min_radius_small = 6

##################
# END user input #
##################

# Define both data and output directories
datdir = os.path.join(base_folder,data_folder,'data')
outdir = os.path.join(base_folder,data_folder,output_folder)

# Create folder if not existing
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Check if data folder is present
if not os.path.exists(datdir):
    raise FileNotFoundError(f'Data path: {datdir} not accessible.')
else:
    print(f'Using data folder: {datdir}...')

# Now find images to process
filelist = [x for x in os.listdir(datdir) if x.endswith('.tif') and x.startswith(img_base)]

# Initialize data dictionary
Datadict = dict()

# Start file loop
n = 0
for fname in sorted(filelist):

    # Counter
    n = n + 1

    # Image counter (for detection figures)
    count = 0

    # Print status
    print(f'Analyzing file: {fname}')

    # 1) Open Image and Background 
    fname_path = os.path.join(datdir,fname)
    img = DIA.open_image_grayscale(fname_path)
    outfile = os.path.join(outdir,'detection',f'{count}.png')
    DIA.save_image(img,outfile)
    count += 1
    if background is not None:
        bg_img_path = os.path.join(datdir,background)
        bkg = DIA.open_image_grayscale(bg_img_path)

    # 2) Substract background
    if background is not None:
        process_image = cv.subtract(img,bkg)
    else:
        process_image = img

    # 3) Rotate and crop image
    process_image = DIA.rotate_image(process_image,angle)
    if Crop is True:
        process_image = process_image[YCoord:YCoord+Height, Xcoord:Xcoord+Width]
    outfile = os.path.join(outdir,'detection',f'{count}.png')
    DIA.save_image(process_image,outfile)
    count += 1

    # Save a copy for later processing
    crop_image = cv.cvtColor(process_image, cv.COLOR_GRAY2BGR)

    # 4) Detect Large (between 35 and 60 pixel radius) bubbles
    print('Large bubbles\n')
    circles_large, count = DIA.detect_bubbles(process_image, noise_large, holes_large, hough_large, count, outdir)
    
    # 5) Remove large from image
    process_image = DIA.remove_detected_circles(process_image, circles_large)
    outfile = os.path.join(outdir,'detection',f'{count}.png')
    DIA.save_image(process_image,outfile)
    count += 1

    # 6) Detect Medium (between 17 and 35 pixel radius) bubbles
    print('Medium bubbles\n')
    circles_medium, count = DIA.detect_bubbles(process_image, noise_medium, holes_medium, hough_medium, count, outdir)

    # 7) Remove medium from image
    process_image = DIA.remove_detected_circles(process_image, circles_medium)
    outfile = os.path.join(outdir,'detection',f'{count}.png')
    DIA.save_image(process_image,outfile)
    count += 1

    # 8) Detect Small (between 6 and 16 pixel radius) bubbles
    print('Small bubbles\n')
    circles_small, count = DIA.detect_small(process_image, max_radius_small,min_radius_small,count)

    # 9) Draw all circles on the detection image
    crop_image = DIA.draw_circles(crop_image, circles_large,  (8, 124, 32))
    crop_image = DIA.draw_circles(crop_image, circles_medium, (255, 0, 255))
    crop_image = DIA.draw_circles(crop_image, circles_small,  (0, 0, 255))

    # 11) Save image
    output_file = os.path.join(outdir,f'detected{n:05d}.png')
    DIA.save_image(crop_image,output_file)

    # Save data in the dictionary
    Datadict[fname] = DIA.save_dictionary(circles_large,circles_medium,circles_small,fname)

    # DIA.show_image(crop_image)
    
# Post processing
#----------------

# First get size (radius) in mm instead of pixels
# TODO

# Plot histograms of bubble sizes
n = 0       
max_radius = 60
bins = np.linspace(0,max_radius,9,dtype=int)
max_count = 0.15
Number = np.zeros(len(Datadict))
for entry in Datadict.keys():

    # Counter
    n = n + 1

    # Check if empty
    if Datadict[entry] is None:
            continue

    # Open a new figure
    plt.figure()

    # Make histogram of bubble sizes
    plt.hist(Datadict[entry][:,2], bins, density=True)

    # Make plot nice
    plt.xlabel('Bubble radius [px]')
    plt.ylabel('Probability [-]')
    plt.title(f'Histogram of bubble radii for file {entry}')
    plt.grid(True)
    plt.xlim((0,max_radius))
    plt.ylim((0,max_count))

    # Save histogram
    plt.savefig(f'{output_folder}/pdf{n:05d}.png',format="png")

    # Get number
    Number[n-1] = Datadict[entry].shape[0]

# Make a plot of (detected) bubble numbers
time = np.arange(0,len(filelist))
plt.figure()

plt.plot(time,Number,'o')

plt.xlabel('Time [min]')
plt.ylabel('Bubble count [-]')
plt.grid(True)
plt.ylim((0,300))

outfig = os.path.join(outdir,'bubble_count.png')
plt.savefig(outfig,format="png")

# Save mat file of analysis
sio.savemat(os.path.join(outdir,'output.mat'), Datadict)