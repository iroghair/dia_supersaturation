#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code generates an image with many bubbles starting from one single bubble.

Last modification: 29/08/2020
Author: Alessandro Battistella
"""

# Imports
import cv2 as cv
import numpy as np

# Functions
import generate_functions as g

# Parameters
N = 200
noise = False

# Image size in pixels
height = 2000
width = 2000

# Open the one_bubble image
single_bubble = cv.imread('data/one_bubble.tif')

# Create a black image
image = np.zeros((height,width,3),np.uint8)

# Bubble size distribution
radii = np.zeros(N)

# Now put all bubbles in the image randomly
for i in range(1,N+1):

    # Generate random single bubble image
    processed_single_bubble, scale = g.generate_single_bubble_random(single_bubble)
    radii[i-1] = np.around(30 * scale)
    
    # Now we place the bubble into the larger image (without overlaps)
    image = g.place_single_bubble(image,processed_single_bubble)

# g.show_image(image)

# Add noise
image = g.generate_noise(image,0.1)
g.show_image(image)

# Write the image
cv.imwrite('data/data1.tif',image)
    
# Save bubble size distribution
np.save('results/generated_distribution.npy', radii)