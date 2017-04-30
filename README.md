**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./output_images/undistort_images_highway.png "Undistorted"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./output_images/flowchart.png "flowchart"
[image8]: ./output_images/color_threshold.png "color_threshold"
[image9]: ./output_images/gradient_threshold.png "gradient threshold"
[image10]: ./output_images/birds_eye_view.png "Birds eye view"
[image11]: ./output_images/poly_fit.png "Fit polynomial"
[image12]: ./output_images/Overlayed_back.png "Overlayed"
[video1]: ./ouput_images/outputproject_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. The basic workflow for lane finding using advanced computer vision techniques is as shown below

![alt text][image7]

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

THe code for this step is present in code blocks [5,6,7] of the ipython notebook advanced_lane_finding.ipynb

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
The code can be found in cells 9 of ipython notebook advanced_lane_finding.ipynb

![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

THe code for the perspective tranform is as shown in code cell 10 of ipython notebook advanced_lane_finding.ipynb . The source and destination points are as chsen below.

This resulted in the following source and destination points:
h = 720
w = 1280

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575,464       | 450,0         | 
| 707,464       | w-450,0      |
| 258,682       | 450,h      |
| 1049,682      | w-450,h        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt_text][image10]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The HSV color space was chosen to process the image to threshold the lane lines. The S and V channel was thresholded to extract the yellow and white lane lines. S and V channels were also subjected to horizontal gradient to extract the lane lines 

![alt text][image8]

The S and V channels are again subjected to gradient thresholds in the horizontal direction to extract the lane lines and the output is as shown below

![alt_text][image9]

The above outputs can be found in code cells 12 and 13 of iPython notebook

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

If in previous frame if polynomial is not fit then a histogram search is done. once polynomial is fit in previous frame, only the vicinity of polynomial is checked for pixels . The code can be found in code cell 15 of ipython notebook advanced_lane_finding.ipynb

![alt_text][image11]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature was implemented using the function get_radius_of_curvature() in helper_function.py
The offset to the center was done using the function car_position_to_center() function in helper_functions.py


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 146 through 168 in my code in `finding_lines_expt.py`
Here is an example of the test image

![alt text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The output video can be found in ./output_images/outputproject_video.mp4
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here the major problem is 
-> The solution is very prone to parameter tuning. Parameters tuned for a use case may not work well another use case
-> The implemented temporal filtering which filters 20 frames of polynomial coeffcients, takes into account a present polynomial fit which differs from previous fits as a bad frame!. This metod works if previous frames are a good fit, else it results in catastropie. 
-> A buffer reset setting needed which reset the averaging buffer when more than a certain number of frames results in different polynomial coeffcients indicating lane shapes have changed 

