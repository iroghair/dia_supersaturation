# Module including all DIA functions for nice code separation

import cv2 as cv
import numpy as np
from skimage import morphology
from skimage import feature

# General image processing functions
#----------------------------

# Open image
def open_image_grayscale(image_name):
    """ This function uses cv2 imread to open a grayscale image
    and checks for errors in the opening process """
    image = cv.imread(image_name,cv.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(('Unable to open image ' + image_name + '. \nExiting...'))
    return image

# Save image 
def save_image(image,name):
    """Save image in a given file name and path"""
    cv.imwrite(name,image)

# Show image 
def show_image(image):
    """ Function to show image until key is pressed"""
    cv.namedWindow('image',cv.WINDOW_NORMAL)
    cv.resizeWindow('image',(1000,1000))
    cv.imshow('image',image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Image manipulation
#--------------------

# Remove small objects (noise)
def remove_small(image,min_size,connectivity):
    """ Removes all the small objects below a minimum size and 
    with a given connectivity with the others. Requires a bool 
    image for conversion """
    return morphology.remove_small_objects(image.astype('bool'), min_size, connectivity).astype('uint8')*255

# Close holes
def close_holes(image,iterations):
    """ This function closes the bubbles to improve detection.
    Still needs to remove some noise so another noise remove 
    is performed. """
    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(3,3))
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=4)
    return image

# Canny edge
def canny_edge(image):
    """ First blur and then perform a canny edge detection """
    image = cv.blur(image,(5,5))
    return feature.canny(image, sigma=3).astype('uint8')*255

# Hough circle transform
def hough_transform(image,param1,param2,minRadius,maxRadius):
    """ This function uses cv2 hough circle transform to detect
    circles. Parameters 1 and 2 don't affect accuracy as such, more reliability.

    Param1 sets the sensitivity: how strong the edges of the circles need to be.
    Too high and it won't detect anything, too low and it will find too much clutter.
    
    Param2 sets how many edge points it needs to find to declare that it's found a circle.
    Too high will detect nothing, too low will declare anything to be a circle. 

    The ideal value of param22 is related to the circumference of the circles. 
    
    Return value to be printed should be an integer. For this reason they are rounded (rint)
    and converted to ints (.astype(int)) """

    # Detect circles
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, minRadius,  
        param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # If circles are found, convert to int
    if circles is not None:
        circles = np.rint(circles).astype(int)
        return circles[0,:] # Remove one additional dimension added for some reason by opencv
    else:
        return None

# Draw circles
def draw_circles(image,circles,color):
    """ Function to draw circles on a given image """
    if circles is not None:
        for i in circles:
            # draw the outer circle
            cv.circle(image, (i[0],i[1]), i[2], color, 2, cv.LINE_AA)
            # draw the center
            cv.circle(image, (i[0],i[1]), 2, (56, 173, 210), 1, cv.LINE_AA)
    return image

# Remove circles
def remove_detected_circles(image, circles):
    """ Draw a black circle over the image, thus actively removing the circles alraedy detected """
    if circles is not None:
        for i in circles:
            cv.circle(image, (i[0],i[1]), i[2], (0, 0, 0), -1, cv.LINE_AA)  # draw a black circle
    return image

# Bubble detection procedure
#----------------------------

def detect_bubbles(image, noise, holes, hough):
    """ This function takes a thresholded and prepared image and performs a detection of bubbles
    within a specific size. Parameters should be adjusted to the specific size. The procedure consists 
    of a series of steps:
        1) Remove noise
        2) Close holes
        3) Canny edge
        4) Hough circle transform
    """

    if noise['min_size'] == 1300:
        n = 2
    elif noise['min_size'] == 920:
        n = 6
    else:
        n = 10

    # Remove noise
    image = remove_small(image, noise['min_size'], noise['connectivity'])
    save_image(image,f'results/detection/{n}.png')

    # Close holes
    image = close_holes(image, holes['iterations'])
    save_image(image,f'results/detection/{n+1}.png')

    # Canny edge
    image = canny_edge(image)
    save_image(image,f'results/detection/{n+2}.png')

    # Detect circles
    circles = hough_transform(image, hough['param1'], hough['param2'], hough['minRadius'], hough['maxRadius'])

    # Print how many bubbles were found
    if circles is not None:
        print(f'Detected: {circles[:,0].size} bubbles\n')
    else:
        print('Detected: 0 bubbles\n')

    return circles

def save_dictionary(circles_large,circles_medium,circles_small, fname):
    """ Function to check if data is empty and store it into a dictionary """

    # Get the array
    circles = np.array([circles for circles in (circles_small, circles_medium, circles_large) if circles is not None])
    
    # Check if empty
    if circles.size == 0:
        circles = None
        print(f'Nothing detected for file {fname}')
    else:
        circles = np.concatenate(circles)

    return circles

# General
#---------

def max_radius(DataDictionary):
    """ Find max radius in the dictionary """
    max = 0
    for key in DataDictionary.keys():
        local_max = np.amax(DataDictionary[key][:,2])
        if max < local_max:
            max = local_max
    
    return max
        
