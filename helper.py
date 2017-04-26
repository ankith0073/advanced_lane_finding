# author: Ankith Manjunath
# Date : 26.04.17
import cv2
import numpy as np
import glob
import pickle
from pathlib import Path

class helper:

    def __init__(self,
                 calib_file,
                 frame,
                 frame_size,
                 src = np.float32([(575,464),(707,464),(258,682),(1049,682)]),
                 dst = np.float32([(450,0),(1280-450,0),(450,720),(1280-450,720)])):
        self.calibration_data_file = calib_file
        self.image = np.zeros_like(frame)
        self.undist = np.zeros_like(frame)
        self.sobel_xbinary = []
        self.mtx = []
        self.dist = []
        self.warped = []
        self.src = src
        self.dst = dst



    #Function to undistort image given the intrinsec and extrinsic matrices
    def undistort_frame(self):
        self.undist = cv2.undistort(self.image, self.mtx, self.dist, None, self.mtx)
        #return undist

    def abs_sobel_thresh(self, orient='x', thresh_min=0, thresh_max=255):

        #1)input is either S or V channel of the image
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(self.image, cv2.CV_64F, 1, 0)
        else:
            sobel = cv2.Sobel(self.image, cv2.CV_64F, 0, 1)

        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)

        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max

        self.sobel_xbinary = np.zeros_like(scaled_sobel)
        self.sobel_xbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1


    def camera_calib(self):
        #check if calibration exists, if it doesnt exist perform calibration and return the calibration matrices
        if not (self.calibration_data_file.is_file()):
            chess_board_squares = [6, 9]
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((chess_board_squares[0] * chess_board_squares[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chess_board_squares[1], 0:chess_board_squares[0]].T.reshape(-1, 2)

            # Arrays to store object points and image points from all the images.
            objpoints = []  # 3d points in real world space
            imgpoints = []  # 2d points in image plane.

            images = glob.glob('camera_cal/calibration*.jpg')

            for idx, fname in enumerate(images):
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, (chess_board_squares[1], chess_board_squares[0]), None)

                # If found, add object points, image points
                if ret == True:
                    objpoints.append(objp)
                    imgpoints.append(corners)

            img_size = (img.shape[1], img.shape[0])
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

            # store the calibration matrices in a file
            dist_pickle = {}
            dist_pickle['mtx'] = mtx
            dist_pickle['dist'] = dist
            pickle.dump(dist_pickle, open(self.calibration_data_file, "wb"))

            self.mtx = []
            self.dist = []
            return
        else:
            dist_pickle = pickle.load(open(self.calibration_data_file, 'rb'))
            self.mtx = dist_pickle['mtx']
            self.dist = dist_pickle['dist']
            return

        #function to perform perspective tranformation
    def perspective_tranformation_birds_eye(self):
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        # Warp the image using OpenCV warpPerspective()
        self.warped = cv2.warpPerspective(self.img, M,
                                     (self.img.shape[1], self.img.shape[0]),
                                     flags=cv2.INTER_LINEAR)









