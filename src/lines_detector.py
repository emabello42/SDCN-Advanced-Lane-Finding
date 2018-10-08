import numpy as np
import cv2
import matplotlib.pyplot as plt
from .camera import Camera
from .basic_blind_search import BasicBlindSearch
from .line import Line

class LinesDetector(object):

    def __init__(self, camera):
        # HYPERPARAMETERS
        self.curvature_tolerance_m = 1000 # maximum difference between both curvatures in meters
        self.parallel_tolerance = 2.0 # maximum difference between a and b coefficients of both curves
        
        
        self.camera = camera
        self.ploty = None #keep the y values for plotting
        self.L = None
        self.R = None
        self.failed_sanity_checks = 0
        self.blind_search = BasicBlindSearch()
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        self.L = Line(self.xm_per_pix, self.ym_per_pix)
        self.R = Line(self.xm_per_pix, self.ym_per_pix)
        self.radius_of_curvature_m = 0

    def draw_lane(self, undist, warped):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        self.L.draw(color_warp, [255,0,0])
        self.R.draw(color_warp, [0,0,255])
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.L.bestx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.R.bestx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.camera.Minv, (undist.shape[1], undist.shape[0])) 
        
        # Write useful information on the final result
        lane_center = (self.R.best_fit[2] - self.L.best_fit[2])/2 + self.L.best_fit[2]
        img_center = undist.shape[1]/2
        offset_from_center_m = np.round((lane_center-img_center)*self.xm_per_pix,2)
        
        cv2.putText(newwarp, "Radius of Curvature = "+str(self.radius_of_curvature_m)+"m", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(newwarp, "Vehicle is "+str(offset_from_center_m)+"m left of center", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result

    def process_frame(self, img):
        undist_img = self.camera.undistort(img)
        try:
            binary_img = self.camera.apply_thresholds(undist_img)
            warped = self.camera.warp(binary_img)
            self.ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
            if not(self.L.allx.any()) or not(self.R.allx.any()) or self.failed_sanity_checks > 3:
                l_allx, l_ally, r_allx, r_ally = self.blind_search.search(warped)
                found = l_allx.any() | r_allx.any()
                if found:
                    if l_allx.any():
                        self.L.allx = l_allx
                        self.L.ally = l_ally
                        self.L.fit_polynomial(self.ploty)
                    if r_allx.any():
                        self.R.allx = r_allx
                        self.R.ally = r_ally
                        self.R.fit_polynomial(self.ploty)
                    if self.sanity_check(warped):
                        self.L.update(self.ploty)
                        self.R.update(self.ploty)
                        self.failed_sanity_checks = 0
                    else:
                        if not(self.L.bestx.any()):
                            if self.L.current_plotx.any():
                                self.L.update(self.ploty)
                            else:
                                raise Exception
                        if not(self.R.bestx.any()):
                            if self.R.current_plotx.any():
                                self.R.update(self.ploty)
                            else:
                                raise Exception
                elif not(self.L.bestx.any()) or not(self.R.bestx.any()):
                    raise Exception
            else:
                self.search_around_poly(warped)
                self.L.fit_polynomial(self.ploty)
                self.R.fit_polynomial(self.ploty)
                if self.sanity_check(warped):
                    self.L.update(self.ploty)
                    self.R.update(self.ploty)
                    self.failed_sanity_checks = 0
                else:
                    self.failed_sanity_checks += 1
            result = self.draw_lane(undist_img, warped)
        except:
            print("There was an error drawing the lane!")
            result = undist_img
        return result


    def search_around_poly(self, binary_warped):
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        self.L.update_search_area(nonzerox, nonzeroy)
        self.R.update_search_area(nonzerox, nonzeroy)

    def sanity_check(self, img_ref):
       ''' Proof the sanity of two lane lines'''

       if self.L == None or self.R == None:
           return False

       if self.L.detected == False or self.R.detected == False:
           return False
       
       # Check that they have similar curvature
       if np.absolute(self.L.radius_of_curvature_m - self.R.radius_of_curvature_m) > self.curvature_tolerance_m:
           return False
       
       # Check that they are separated by approximately the right distance
       # horizontally
       if not(3.7 < (self.R.current_fit[2]*self.xm_per_pix - self.L.current_fit[2]*self.xm_per_pix) < 5.6):
           return False

       # Check that they are roughly parallel
       a = np.absolute(self.R.current_fit[0] - self.L.current_fit[0])
       b = np.absolute(self.R.current_fit[1] - self.L.current_fit[1])
       if a > self.parallel_tolerance or b > self.parallel_tolerance:
           return False
       
       self.radius_of_curvature_m = np.round((self.L.radius_of_curvature_m + self.R.radius_of_curvature_m)/2,2)
       return True
