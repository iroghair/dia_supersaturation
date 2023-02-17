import cv2 as cv
import numpy as np
from numpy.random import default_rng

# Show image 
def show_image(image):
    """ Function to show image until key is pressed"""
    cv.namedWindow('image',cv.WINDOW_NORMAL)
    cv.resizeWindow('image',(1000,1000))
    cv.imshow('image',image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def generate_single_bubble_random(image):
    """ Process a single bubble image and randomly
        rotate and resize it """
    
    # Generate random scale factor 
    # (between 2 and 0.2 because we have bubbles of radius (6,60) and
    # our bubble image has radius 30
    # random_scale = (1.8-0.5) * np.random.rand(1) + 0.2
    random_scale = default_rng().normal(1,0.3)

    # Now resize the image to get a random radius (do we really want that? TODO maybe necessary only flip image because we always want to have illumination from the sides)
    image = cv.resize(image, (0,0), fx=random_scale, fy=random_scale)

    # Now we turn the bubble to some angle between 0 and 360 (361 is not included!)
    angle = np.random.randint(361)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)

    return result, random_scale

def find_index(image,bubble):
    """ Give index which don't overlap """

    # First get image size in pixels
    x,y = image.shape[0:2]

    # Then get a random position within the image
    x_loc = np.random.randint(x - bubble.shape[0] + 1)
    y_loc = np.random.randint(y - bubble.shape[1] + 1)

    # Find index of image in the picture
    x_start = x_loc
    x_end = x_loc + bubble.shape[0]
    y_start = y_loc
    y_end = y_loc + bubble.shape[1]

    # Check if the considered block is fully black (all zeros)
    all_zero = not image[x_start:x_end,y_start:y_end].any()

    # If yes, return the given values, else generate again
    if all_zero is True:
        return x_start,x_end,y_start,y_end
    else:
        return find_index(image,bubble)

def place_single_bubble(image,bubble):
    """ Place single bubble without overlaps in the larger image """

    # Get the index for the bubble to be placed
    x_start,x_end,y_start,y_end = find_index(image,bubble)

    # Place image in the given position
    image[x_start:x_end,y_start:y_end] = bubble

    # Return the processed image
    return image

def generate_noise(image,probability):
    """ generate random white noise in the image """
    
    # Set white color for both B&W and RGB image
    if len(image.shape) == 2:
        white = 255
    else:
        white =  np.array([255, 255, 255], dtype='uint8')

    # Generate random numbers
    probs = np.random.random(image.shape[:2])
    image[probs < (probability / 2)] = white

    return image