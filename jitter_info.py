# author: Ankith Manjunath
# Date : 26.04.17
import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        #self.current_fit = [np.array([False])]
        #self.current_fit = np.zeros(shape = [3,5] , dtype = np.float32)
        self.current_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        # polynomial coefficients for the most recent fit
        # self.current_fit = [np.array([False])]
        # self.current_fit = np.zeros(shape = [3,5] , dtype = np.float32)
        self.current_fit = None
        #polynomial coefficients for the most recent fit in global coordinates
        self.fit_world = None

        #stor info about disatance to center
        self.dist_to_center = None