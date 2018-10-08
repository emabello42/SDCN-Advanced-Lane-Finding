## Advanced Lane Finding Project
### Udacity Self Driving Car Engineer Nanodegree - Project 2

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

[image1]: ./output_images/undistort_img.jpg "Undistorted"
[image2]: ./output_images/undistorted_test.jpg "Road Transformed"
[image3]: ./output_images/binary_test.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[video1]: ./output_videos/out_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the lines 20 through 50 of the file called `src/camera.py` (method `calibrate()` in the class `Camera`).  An example can be seen in the second cell of the IPython notebook `P2`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` (see lines 52 through 55 of the same file ) function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 70 through 109 in `src/camera.py`: functions `apply_thresholds()`, `color_threshold()`, `grad_threshold()`).  An example can be seen in the 4th cell of the IPython notebook `P2`

The best combination of color and gradient thresholds that worked for me, is using the grayscale representation to calculate the gradient through the *x* axis to detect vertical lines, and the HSV color representation to find the yellow lane lines using the V and H channels (because in grayscale this color almost disappears). Moreover, the white pixels were also combined with the gradient threshold to improve the chances of finding white lane lines.

Here's an example of my output for this step:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes the functions `warp()` and `generate_perspective_transform()`, that appear in lines 111 through 145 in the file `src/camera.py`. An example is in the 5th code cell of the IPython notebook. 
The `generate_perspective_transform()` function calculates the transformation matrix M and its inverse Minv. The `warp()` function takes as input an image (`img`) and warps it using M, so that we can have a birds-eye view of the undistorted image. I chose to hardcode the source and destination points in the following manner:

```python
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
```

This resulted in the following source and destination points:
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 195, 0        | 
| 700, 460      | 1110, 0       |
| 1110, 719     | 1110, 719     |
| 195, 719      | 195, 719      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After applying a perspective transform to the binary image, the detection of lane line pixels takes place in the `search()` method of the class `BasicBlindSearch` (file `src/basic_blind_search.py`), which returns the pixel positions of the left and right lane lines. This method searches for lane line pixels using a sliding window approach, which starts by taking a histogram of the bottom half of the image and uses the peaks of the left and right halves of the histogram as the starting points for the left and right lane lines. The procedure continues sliding a window through both lane line pixels to the top of the binary image (the horizontal position of this sliding window is updated based on the mean position of the pixels inside it).

Then the `fit_polynomial()` method of the class `Line` (file `src/line.py`) receives the pixel positions of a lane line and fits a 2nd order polynomial using the `numpy.polyfit()` function. The figure following figure depicts the result:


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

 The `fit_polynomial()` method of the class `Line` also calculates the radius of curvature of each lane line (see lines 50 through 52 in `src/line.py`). The radius of curvature of the lane is calculated as the average of both lane lines (see line 139 in  `src/lines_detector.py`).

The position of the of vehicle is calculated in the lines 46 through 48 in `src/lines_detector.py`. It uses the intercepts of both polynomials (i.e. the *y* value where the curve intercepts the *x* axis) and finds the middle position between them, which corresponds to the middle of the lane. Then the difference between such middle and the middle of the image (car position) is calculated and converted from pixels to meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 28 through 54 in my code in `src/lines_detector.py` in the function `draw_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/out_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach that I followed to detect lane lines in video, is keeping two separate instances of the class Line, for the left and right line, that keep track of the last 5 lines that were correctly detected. To check if a lane line was correctly detected, a `sanity_check()` method was implemented (see lines 115 through 140 in `src/lines_detector.py`), which checks that the left and right lane lines have approximately the same radius of curvature, are separated by approximately the same distance and are roughly parallel. Then, the average of the polynomial coefficients of the lines detected are used to draw the lane area in the output video.

On the other hand, to optimize the detection, once both lane lines were detected using a blind search (see point 4. above), I search for lane line pixels in the following video frames around the curves previously fitted. But if the sanity checks fails until three times, a blind search of lines pixels starts again.

During the implementation of my pipeline, I found particularly difficult to find the best combination of color and gradient thresholds (see point 2 above) that best highlight the lane lines and the blind search of lane line pixels.

Although the current implementation works quite well for the project video, the pipeline is still experimenting some problems with the challenge video. Probably this is due to the detection of false positive lane lines during the blind search, because from the binary images more than one possible lane line can be found on each side of the image. For this reason, I think that a possible improvement could be to take more than one peak in the left and right halves of the histogram, and from them starting a separately search. Then, a sanity check procedure can be used to determine the two lines that truly correspond to the lane. 

Moreover, to face the large curves shown in the harder challenge video, the blind search method could be further improved if the sliding window was able to search line pixels from the bottom image to the left or right sides.