# author: Ankith Manjunath
# Date : 23.04.17

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import imageio
import pickle
from pathlib import Path
from helper_functions import *
from jitter_info import Line

images = glob.glob('./test_images/straight_lines*.jpg')
input_video = 'project_video.mp4'
output_video_folder = './'
out_color_thresolded = imageio.get_writer(output_video_folder + 'output' + input_video  , fps=20)
reader = imageio.get_reader('./' + input_video)


calibration_data_file = Path('/wide_dist_pickle.p')
dist_pickle = pickle.load(open('./camera_cal/wide_dist_pickle.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']


#kernel size definition
ksize = 3

#threshold for x and y
thresh_x_min = 30
thresh_x_max = 150

h = 720
w = 1280
# define source and destination points for transform
src = np.float32([(575,464),
                  (707,464),
                  (258,682),
                  (1049,682)])
dst = np.float32([(450,0),
                  (w-450,0),
                  (450,h),
                  (w-450,h)])

#HSV threshold for color thresholding
threshold = [[0,70],
             [150,255],
             [220,255]]

#create objects of the left and right lane
left_info = Line()
right_info = Line()

for i, img in enumerate(reader):
    #frame = cv2.imread(img)
    undistort = undistort_frame(img, mtx, dist)

    M = cv2.getPerspectiveTransform(src, dst)
    #get the inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst,src)
    # Warp the image using OpenCV warpPerspective()
    frame = img
    warped = cv2.warpPerspective(undistort, M,
                                 (frame.shape[1], frame.shape[0]),
                                 flags=cv2.INTER_LINEAR)

    hsv_img = cv2.cvtColor(warped, code=cv2.COLOR_BGR2HSV)

#color thresholding
    # separate color channels
    h = hsv_img[:, :, 0]
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]

    h_threshold = np.zeros_like(h)
    h_threshold[np.logical_and(h > threshold[0][0], h < threshold[0][1])] = 1

    s_threshold = np.zeros_like(s)
    s_threshold[np.logical_and(s > threshold[1][0], s < threshold[1][1])] = 1

    v_threshold = np.zeros_like(s)
    v_threshold[np.logical_and(v > threshold[2][0], v < threshold[2][1])] = 1

    h_s_v_combined_mask = np.logical_or(s_threshold, v_threshold)

    #gradient thresholding
    #gradient threshold in horizontal direction the saturation channel
    gradx_s = abs_sobel_thresh(s,
                               orient='x',
                               thresh_min=thresh_x_min,
                               thresh_max=thresh_x_max)

    # gradient threshold in horizontal direction the value channel
    gradx_v = abs_sobel_thresh(v,
                               orient='x',
                               thresh_min=thresh_x_min,
                               thresh_max=thresh_x_max)

    #combine the binary masks
    combined_gradient = np.logical_or(gradx_s, gradx_v)

    #combine color and gradient masks
    combined_gradient_color = np.logical_or(combined_gradient, h_s_v_combined_mask)

    binary_warped = np.zeros_like(v)
    binary_warped[combined_gradient_color == True] = 1

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    if i == 0:
        poly_fit_image, left_fit, right_fit = fit_first(binary_warped, left_info, right_info)

    if i != 0:
        poly_fit_image, left_fit, right_fit = fit(binary_warped, left_info, right_info)

    #radius of curvature of left lane
    get_radius_of_curvature(left_info, 719)
    get_radius_of_curvature(right_info, 719)

    dist_to_center(left_info, np.max(ploty))
    dist_to_center(right_info, np.max(ploty))

    position_offset = car_position_to_center(left_info, right_info)

    left_fitx = left_info.current_fit[0] * ploty ** 2 + left_info.current_fit[1] * ploty + left_info.current_fit[2]
    right_fitx = right_info.current_fit[0] * ploty ** 2 + right_info.current_fit[1] * ploty + right_info.current_fit[2]
    #draw the lane back to the original video feed
    # Create an image to draw the lines on
    color_warp = np.zeros_like(warped).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistort, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    out_color_thresolded.append_data(result)



out_color_thresolded.close()

