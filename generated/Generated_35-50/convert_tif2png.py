#!/usr/bin/env python3
""" Quick and fast tif to png converter """

import cv2 as cv

# File name
fname = "data/data1.tif" 

# Converted name 
if ".tif" in fname:
    fname = fname.replace(".tif",".png")
else:
    raise FileNotFoundError('The given file: ' + fname + ' is not a .tif image') 

# Open image
image = cv.imread(fname,cv.IMREAD_GRAYSCALE)

# Save the tif into pdf
cv.imwrite(fname,image)

