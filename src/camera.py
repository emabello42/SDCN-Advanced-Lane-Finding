import glob
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class CalibrationException(Exception):
    pass


class Camera(object):

    def __init__(self, nx=9, ny=6):
        self.dist = None
        self.mtx = None
        self.M = None
        self.Minv = None
    
    def calibrate(self, nx=9, ny=6, path_pattern="../camera_cal/*.jpg"):
        '''Calibrate camera using chessboard images'''
        # prepare object points
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(path_pattern)

        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        img_size = (img.shape[1], img.shape[0])

        # Do camera calibration given object points and image points
        if len(imgpoints) > 0:
            ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        else:
            raise CalibrationException("image points not found")

    def undistort(self, image):
        '''Undistort image using calibration matrix and distortion coefficients'''
        dst = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        return dst

    def save_calibration(self, file_name="../camera_cal/calibration_info.p"):
        '''Save the camera calibration result for later use'''
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist
        pickle.dump( dist_pickle, open( file_name, "wb" ) )

    def load_calibration(self, file_name="../camera_cal/calibration_info.p"):
        '''Read the camera matrix and distortion coefficients from a file'''
        dist_pickle = pickle.load( open( file_name, "rb" ) )
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]

    def apply_thresholds(self, img):
        '''Apply color and gradient thresholds and return a binary image'''
        img = np.copy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Convert to grayscale
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # Convert to HSV color space
        
        white, yellow = self.color_threshold(hsv) # Get masks for white and yellow lane lines
        grad_thr = self.grad_threshold(gray) # Get masks for gradients

        combined = np.zeros_like(gray)
        # The gradient thresholds work better for white lane lines, while
        # yellow lane lines can be better detected using only color thresholds
        combined[((grad_thr==1) & (white==1)) | (yellow==1)] = 1
        return combined
    
    def color_threshold(self, hsv):
        '''Apply color thresholds for white and yellow'''
        h_channel = hsv[:,:,0]
        s_channel = hsv[:,:,1]
        v_channel = hsv[:,:,2]

        white = np.zeros_like(s_channel)
        white[(s_channel <= 20)] = 1

        yellow = np.zeros_like(h_channel)
        yellow[(v_channel >= 170) & (v_channel <= 255) & (20<=h_channel) & (h_channel <=70)] = 1
        
        return white, yellow

    def grad_threshold(self, gray):
        '''Apply gradient thresholds'''
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobelx)
        sxbinary[(scaled_sobelx >= 20) & (scaled_sobelx <= 100)] = 1

        return sxbinary
    
    def generate_perspective_transform(self, src_filename="../test_images/straight_lines1.jpg"):
        '''
        Generate a perspective transform matrix and its inverse based on a image
        with straight lane lines
        '''

        img = mpimg.imread(src_filename)
        undist_img = self.undistort(img)
        img_size = (img.shape[1], img.shape[0])
    
        # source points
        top_margin = 100
        bottom_margin= 470
        sp0 = (img_size[0]//2-top_margin+40, img_size[1]//2+100)
        sp1 = (img_size[0]//2+top_margin-40, img_size[1]//2+100)
        sp2 = (img_size[0]//2+bottom_margin, img_size[1]-1)
        sp3 = (img_size[0]//2-bottom_margin+25, img_size[1]-1)
        src = np.float32([sp0, sp1, sp2, sp3])
        
        # destination points
        dp0 = (sp3[0],0) 
        dp1 = (sp2[0],0) 
        dp2 = (sp2[0],img_size[1]-1)
        dp3 = (sp3[0],img_size[1]-1) 
        dst = np.float32([dp0, dp1, dp2, dp3])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        
        return src, dst
    
    def warp(self, img):
        ''' Warp image'''
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.M, img_size)
        return warped
