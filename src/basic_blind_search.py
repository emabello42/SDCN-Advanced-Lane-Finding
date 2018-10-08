from .line import Line
import numpy as np
import cv2

class BasicBlindSearch(object):

    def __init__(self):
        # HYPERPARAMETERS
        self.nwindows = 9 # Choose the number of sliding windows
        self.margin = 150 # Set the width of the windows +/- margin
        self.minpix = 50 # Set minimum number of pixels found to recenter window

    def search(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//self.nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx = np.array([])
        lefty = np.array([])
        rightx = np.array([])
        righty = np.array([])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            # Find the four below boundaries of the window
            win_xleft_low = leftx_current-self.margin 
            win_xleft_high = leftx_current+self.margin 
            win_xright_low = rightx_current-self.margin 
            win_xright_high = rightx_current+self.margin
    
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            if len(good_left_inds) > 0:
                left_lane_inds.append(good_left_inds)
            if len(good_right_inds) > 0:
                right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window
            #(`right` or `leftx_current`) on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        if len(left_lane_inds) > 0:
            left_lane_inds = np.concatenate(left_lane_inds)
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]

        if len(right_lane_inds) > 0:
            right_lane_inds= np.concatenate(right_lane_inds)
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty
