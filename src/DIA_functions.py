# Module including all DIA functions for nice code separation

import os
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

# Rotate image
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result

# Image manipulation
#--------------------

def blur_image(image):
    """ Blur the image to remove some noise """
    image = cv.medianBlur(image,7)
    return image

def threshold(image):
    """ Do some thresholding operations """
    # ret,image = cv.threshold(image,150,255,cv.THRESH_BINARY)
    # image = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv.THRESH_BINARY,151,-140)
    # show_image(image)
    gamma = 0.125
    image = adjust_levels(image, gamma)    
    image = cv.medianBlur(image,5)
    return image

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

def adjust_levels(img, gamma):
    """ Adjust levels with a given gamma and threshold the resulting image """
    
    # TODO might be nice to have levels in input
    inBlack  = 0
    inWhite  = 255
    inGamma  = gamma
    outBlack = 0
    outWhite = 255

    # Adjust levels
    img = np.clip( (img - inBlack) / (inWhite - inBlack), 0, 255 )                            
    img = ( img ** (1/inGamma) ) *  (outWhite - outBlack) + outBlack
    img = np.clip( img, 0, 255).astype(np.uint8)

    # Threshold
    img = cv.threshold(img, 1 , 255, cv.THRESH_BINARY)[1]

    # Return
    return img

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
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, minDist=2*minRadius,  
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
    """ Draw a black circle (+5 pixels radius) over the image, thus actively removing the circles alraedy detected """
    if circles is not None:
        for i in circles:
            cv.circle(image, (i[0],i[1]), i[2] + 10, (0, 0, 0), -1, cv.LINE_AA)  # draw a black circle
    return image

# Bubble detection procedure
#----------------------------

def get_centroid(contours):
    """ Get the centroid of a contour using the moments"""

    # Preallocate memory
    centroids = np.zeros((len(contours),2))

    # Iterate over contours
    i = -1
    mask = np.ones(len(centroids),dtype=bool)
    for contour in contours:

        # Increase counter
        i = i + 1

        # Find moments
        M = cv.moments(contour)

        # Find centroid
        if M['m00'] == 0:   
            # Entries with no moments (basically unremoved noise) will be removed later
            centroids[i,0] = 0
            centroids[i,1] = 0
            mask[i] = False
        else:
            centroids[i,0] = int(M['m10']/M['m00'])
            centroids[i,1] = int(M['m01']/M['m00'])

    return centroids[mask]

def distance(point_1,point_2):
    """ Distance between two points (2D) """
    return np.linalg.norm(point_1 - point_2)

def find_partner(centroids,i,detected,max_radius,min_radius_small):
    """ Find the partner of i for the circle """

    tolerance = min_radius_small
    partner = None

    # Check if edge already counted
    if detected[i] == True:
        return partner

    # Start loop
    j = i
    for c_2 in centroids[i+1:]:
        
        # Index
        j = j + 1

        # Check if already detected
        if detected[j] == True:
            continue

        # Calculate distance
        d = distance(c_2,centroids[i])

        # Check if they are close enough
        if (d < max_radius*2 and d > min_radius_small*2):

            # Check if they are aligned horizontally
            if np.abs(centroids[i,1] - c_2[1]) <= tolerance:
                partner = j
                break

    return partner

def get_circles_from_pairs(pairs,centroids):
    """ Get circles from the two contour pairs """

    # Init circles
    circles = []

    # Loop all over the pairs
    for p in pairs:

        # Extract centroids
        c_1 = centroids[p[0]]
        c_2 = centroids[p[1]]

        # Center of circle
        cx = (c_1[0] + c_2[0]) / 2
        cy = (c_1[1] + c_2[1]) / 2

        # Radius of circle  
        # TODO usually the centroid of the area is about a couple of pixels short of the actual radius, but we should use the contours for distance determination (that's why I add 3 here)
        r = distance(c_1,c_2)/2 + 3

        # Plug it in circles
        circles.append([cx, cy, r])

    return np.array(circles)

def get_pairs(contours,max_radius,min_radius_small):
    """ From contours get pairs of bubble edges and return the pairs """

    # Get centroids
    centroids = get_centroid(contours)

    # Make a list of pairs
    pairs = []

    # Sort centroids from left to right
    centroids = centroids[centroids[:,0].argsort()]

    # Make a mask and start loop
    detected = np.zeros(len(centroids),dtype=bool)
    i = -1
    for c_1 in centroids:

        # Increase counter
        i = i + 1

        # Detect the possible circle
        partner = find_partner(centroids,i,detected,max_radius,min_radius_small)

        # Check if something got detected otherwise move on
        if partner is None:
            continue

        # Add to the list
        pairs.append([i,partner])

        # Remove partner from count
        detected[partner] = True

    return pairs,centroids

def detect_small(image, max_radius,min_radius_small,count):
    """ Test function for detection of extremely small bubbles """

    # Blur and Threshold
    blur = blur_image(image)
    image = threshold(blur)
    save_image(image,f'results/detection/{count}.png')
    count += 1

    # Find contours
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Get pairs from contours
    pairs,centroids = get_pairs(contours,max_radius,min_radius_small)

    # Print pairs
    for c in centroids: 
        cv.circle(image,(c[0].astype(int),c[1].astype(int)), 2, (56, 173, 210), 1, cv.LINE_AA)

    save_image(image,f'results/detection/{count}.png')
    count += 1

    # Now get circle from pairs
    circles = get_circles_from_pairs(pairs,centroids).astype(int)

    # Print how many bubbles were found
    if len(circles) == 0:
        print('Detected: 0 bubbles\n')
        return None, count
    else:
        print(f'Detected: {circles[:,0].size} bubbles\n')

    # Return detected circles
    return circles, count

def detect_bubbles(image, noise, holes, hough, count, outdir):
    """ This function takes a thresholded and prepared image and performs a detection of bubbles
    within a specific size. Parameters should be adjusted to the specific size. The procedure consists 
    of a series of steps:
        1) Remove noise
        2) Close holes
        3) Canny edge
        4) Hough circle transform

    """

    # Threshold image
    image = cv.medianBlur(image,11)
    image = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,71,-20)

    # Save image after blur
    outfile = os.path.join(outdir,'detection',f'{count}.png')
    save_image(image,outfile)
    count += 1

    # Remove noise
    image = remove_small(image, noise['min_size'], noise['connectivity'])
    outfile = os.path.join(outdir,'detection',f'{count}.png')
    save_image(image,outfile)
    count += 1

    # Close holes
    image = close_holes(image, holes['iterations'])
    outfile = os.path.join(outdir,'detection',f'{count}.png')
    save_image(image,outfile)
    count += 1

    # Canny edge
    image = canny_edge(image)
    outfile = os.path.join(outdir,'detection',f'{count}.png')
    save_image(image,outfile)
    count += 1

    # Detect circles
    circles = hough_transform(image, hough['param1'], hough['param2'], hough['minRadius'], hough['maxRadius'])

    # Print how many bubbles were found
    if circles is not None:
        print(f'Detected: {circles[:,0].size} bubbles\n')
    else:
        print('Detected: 0 bubbles\n')

    return circles, count

def save_dictionary(circles_large,circles_medium,circles_small, fname):
    """ Function to check if data is empty and store it into a dictionary """

    # Get the array
    circles = np.array([circles for circles in (circles_small, circles_medium, circles_large) if circles is not None], dtype=object)
    
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
        if DataDictionary[key] is None:
            continue
        else:
            local_max = np.amax(DataDictionary[key][:,2])
            if max < local_max:
                max = local_max
        
    return max
        
