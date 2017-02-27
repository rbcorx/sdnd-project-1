#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

img_dir = "test_images/"
img_name = "solidWhiteCurve.jpg"

image = mpimg.imread(img_dir + img_name)

# argument dictionaries with an extra entity "p_order" which depicts 
#     order of arguments by key to be passed in the function, any non 
#     key value is passed as it is
#
# "image" key is special and indicates the place of image in the positional args

# region of interest args
a_reg = {
    "vertices": np.array([[
                    (image.shape[1]//2+10, image.shape[0]//2+40), 
                    (image.shape[1]//2-10, image.shape[0]//2+40), 
                    (50, image.shape[0]), 
                    (image.shape[1]-50, image.shape[0]), 
                 ]], dtype=np.int32),
    
    "a_order": ["image", "vertices",],
}

# gaussian blur args
a_gblur = {
    "kernel_size": 5,
    
    "a_order": ["image", "kernel_size",],
}

# canny edge detection args
a_canny = {
    "low_threshold": 50,
    "high_threshold": 150, 
    
    "a_order": ["image", "low_threshold", "high_threshold"],
}

# houghline detection args  
a_hough = {
    "rho": 1,
    "theta": np.pi/180,
    "threshold": 1,
    "min_line_length": 1,
    "max_line_gap": 1,
    
    "a_order": ["image", "rho", "theta", "threshold", "min_line_length", "max_line_gap"],
}
    
def call_func(func, args, img=None):
    """
    This will process the argument dictionary and call 
    the function using generated lists of positional arguments
    as per the "a_order" value
    
    func: function to be called
    args: arguments to be passed to the function
    img: optional image to be passed
    """
    if not "a_order" in args:
        return func(**args)
    
    gen_args = []
    for arg in args["a_order"]:
        if type(arg) == str and arg in args:
            gen_args.append(args[arg])
        elif(arg == "image"):
            gen_args.append(img)
        else:
            gen_args.append(arg)
    print("lol", gen_args)
    return func(*gen_args)

def pipeline(image):
    
    # greying an image copy
    grey = grayscale(np.copy(image))
    plt.imshow(grey)
    
    # performing gaussian blur
    blurred = call_func(gaussian_blur, a_gblur, grey)
    plt.imshow(blurred)
    
    # performing canny edge detection
    edge = call_func(canny, a_canny, blurred)
    plt.imshow(edge)
    
    # selecting area of interest
    mask = call_func(region_of_interest, a_reg, edge)
    plt.imshow(mask)
    
    
    # performing houghlines selection
    lines = call_func(hough_lines, a_hough, mask)
    plt.imshow(lines)
    plt.show()

pipeline(image)


"""
initial, all 1

    "rho": 10,
    "theta": np.pi/10,
    "threshold": 50,
    "min_line_length": 1,
    "max_line_gap": 50,

connection found~

"rho": 2,
    "theta": np.pi/50,
    "threshold": 20,
    "min_line_length": 10,
    "max_line_gap": 20,

    nice

"rho": 1,
    "theta": np.pi/180,
    "threshold": 20,
    "min_line_length": 10,
    "max_line_gap": 10,


"""
