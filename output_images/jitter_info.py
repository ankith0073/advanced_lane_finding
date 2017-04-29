# author: Ankith Manjunath
# Date : 26.04.17
import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self,n, size_of_x_points):
        self.n     = n
        self.ploty = np.linspace(0, size_of_x_points - 1, size_of_x_points)

        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = np.zeros(shape = [size_of_x_points, n] , dtype = np.float32)

        #average x values of the fitted line over the last n iterations
        self.bestx = None

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

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
        self.current_fit = np.zeros(shape = [3,n] , dtype = np.float32)

        #self.current_fit = None
        #polynomial coefficients for the most recent fit in global coordinates
        self.fit_world = None

        #stor info about disatance to center
        self.dist_to_center = None

        #number of columns to average
        self.n_avg = None

        #keep a counter for error frames
        self.error_frames = None

    def update_xfitted(self):
        self.recent_xfitted[:,0] = self.current_fit[0,0] * self.ploty ** 2 + self.current_fit[1,0] * self.ploty + self.current_fit[2,0]
        #find the number of valid rows for averaging
        self.n_avg = np.count_nonzero(self.current_fit[2, :])



    def update_circ_buf(self):
        for i in range(1,self.n):
            self.recent_xfitted[:,self.n - i] = self.recent_xfitted[:,self.n - i - 1]
            self.current_fit[:, self.n - i] = self.current_fit[:, self.n - i - 1]
        self.recent_xfitted[:, 0] = np.zeros(self.recent_xfitted.shape[0] )
        self.current_fit[:,0] = np.zeros(3)

    def get_best_fit(self):
        self.best_fit = np.sum(self.current_fit, axis=1)/self.n_avg
        self.bestx = self.best_fit[0] * self.ploty ** 2 + self.best_fit[1] * self.ploty + self.best_fit[2]




