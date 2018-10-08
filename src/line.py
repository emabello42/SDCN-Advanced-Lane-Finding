import numpy as np

# Define a class to receive the characteristics of each line detection
class Line(object):
    
    def __init__(self, xm_per_pix, ym_per_pix):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = np.array([]) 
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([])
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([])
        # x values for plotting of the most recent fit
        self.current_plotx = np.array([])
        # y values for plotting of the most recent fit
        self.current_ploty = np.array([])
        #radius of curvature of the line in meters
        self.radius_of_curvature_m = None 
        #x values for detected line pixels
        self.allx = np.array([]) 
        #y values for detected line pixels
        self.ally = np.array([])
        
        self.ym_per_pix = ym_per_pix # meters per pixel in y dimension
        self.xm_per_pix = xm_per_pix# meters per pixel in x dimension
        
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        self.margin = 50

    def draw(self, img, color):
        img[self.ally, self.allx] = color


    def fit_polynomial(self, ploty):
        '''Fit a second order polynomial'''
        try:
            self.current_ploty = ploty
            self.current_fit = np.polyfit(self.ally, self.allx, 2)
            
            # Generate x values for plotting
            self.current_plotx = self.current_fit[0]*ploty**2 + self.current_fit[1]*ploty + self.current_fit[2]
            self.detected = True
            
            #Calculate the curvature of the current fit in meters
            fit_cr = np.polyfit(ploty*self.ym_per_pix, self.current_plotx*self.xm_per_pix, 2)
            y_eval = np.max(ploty)*self.ym_per_pix
            self.radius_of_curvature_m = ((1+(2*fit_cr[0]*y_eval+fit_cr[1])**2)**(3/2))/np.absolute(2*fit_cr[0])
        except:
            # Avoids an error if `current_fit` is still none or incorrect
            print('The function failed to fit a line!')
            self.detected = False
    


    def update(self, ploty):
        '''Update the average using the last 5 correct detections'''
        self.recent_xfitted = self.recent_xfitted[-4:]
        self.recent_xfitted.append(self.current_plotx)
        self.bestx = np.average(self.recent_xfitted, axis=0)
        self.best_fit = np.polyfit(ploty, self.bestx, 2)


    def update_search_area(self, nonzerox, nonzeroy):
        '''
        Update the search area, i.e. the pixels that are going
        to be used to fit a new polynomial
        '''
        lane_inds = ((nonzerox >= self.best_fit[0]*nonzeroy**2+self.best_fit[1]*nonzeroy+self.best_fit[2]-self.margin) &
                    (nonzerox < self.best_fit[0]*nonzeroy**2+self.best_fit[1]*nonzeroy+self.best_fit[2]+self.margin)).nonzero()[0]

        # Extract line pixel positions
        self.allx = nonzerox[lane_inds]
        self.ally = nonzeroy[lane_inds]
