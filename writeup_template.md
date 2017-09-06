**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/orig_car_image.jpg
[image2]: ./output_images/hog_car_image.jpg
[image3]: ./output_images/orig_noncar_image.jpg
[image4]: ./output_images/hog_noncar_image.jpg
[image5]: ./output_images/detected_car.jpg
[image6]: ./output_images/sliding_window_image.jpg
[image7]: ./output_images/heatmap.jpg
[image8]: ./output_images/thresholded_heatmap.jpg
[image9]: ./output_images/gray_bounded_box_heatmap.jpg
[image10]: ./output_images/overlay_final_bounding_box.jpg
[image11]: ./output_images/test_result_bounded_box_images.jpg
[video1]: ./project_video.mp4

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for extracting HOG features from an image is defined by the method get_hog_features.
First, I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image3]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and choose based upon the performance of the SVM classifier produced using them. Most of the time, the accuracy hovered around 94% to 98%. "YUV" and "LUV" seemed to be the two best color spaces to tinker with based on the resulting accuracy. "YUV" on average came out on top compared to that of "LUV" although sometimes they are very close. 

The other parameters were chosen as follows:
- 11 orientations, 16 pixels per cell, 2 cells per block, and ALL channels of the colorspace.

Orientation parameter was tinkered as the project went on. At first, 16 was chosen, then 10, but 11 somehow is the only orientation that works out as I am expecting bounding box around the detected car to be drawn. The other orientation number resulted in either boxes drawn for non-car locations or no box drawn at all.

2 cells per block and 16 pixels per cell was arbitrarily chosen since the difference between the choices are not that visible. If there is, seems to be the speed of execution which is not that important at this point.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with the default classifier parameters and using HOG features (without spatial intensity or channel intensity histogram features) with resulting accuracy of 98.34% under 1 second runtime.

![alt text][image2]
![alt text][image4]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I adapted the method find_cars from Udacity's lesson materials. The methos is called "find_cars". The method combines HOG feature extraction with a sliding window search. The HOG features are extracted from entire frame and then these full-image features are subsampled to 32x32px and then fed to the classifier. The method classifies based on prediction on its HOG features for each window region. The return value is a list of rectangle/squares objects for each windows that has a car prediction.

First attempt with single window results in the following image:

![alt text][image5]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As we can see on examples above, the classifier only successfully found both car the test images. The size of the bounding box is however only 1/4-1/2 of the car. This can potentially spell trouble later on. To accomodate this and also to ensure elimination of the cars on the side of the highway from being detected, heatmap filtering was used (methods are called'add_heat' and 'apply_threshold'). On top of that, many windows were deployed such that the cars on the same lane direction get many more overlapping windows detecting cars. Ystart and ystop parameters were tried again and again. The concept was simple, keep ystart the same for each scale and try on the increment of 25 for ystop and see if the bounding box will still be drawn.

With many sliding windows, the image looks like the following:
![alt text][image6]

Getting the heatmap from using many windows search:
![alt text][image7]

Filtering by putting threshold = 1 on the heatmap:
![alt text][image8]

Gray drawing of the filtered heatmap:
![alt text][image9]

Draw the bounding box by tracing the shape of the filered heatmap result:
![alt text][image10]

Runnig the method above for all the test images:
![alt text][image11]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code is identical to the code for processing a single image described above, just ran over many frames. Filter of false positives was implemented for the single image as well. Basically, many trials were taken to determine collection of ystart and ystop parameters that got fed into collecting the rectangles generated by find_cars method. When the ystart and ystop are too big, false positives were seen. But so far, after determining these ystart and ystop params, the classifier works really well, at least for the given project_video.mp4.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This project pipeline is still pretty slow in speed to accomodate real time usage. I am using GTX 1080 for this project and the time to generate video and so on are noticeable.

The pipeline is probably most likely to fail in cases where vehicles or its HOG feature are not similar to those in the training dataset. Extremely bright or dark conditions seems to become a big impact for hog values as well. In that case, the pipeline will most likely fail too. Oncoming cars from the other lanes will occasionally be detected and it definitely can spell trouble as it cannot differ between cars that are going in the same direction and those ones who are not.

To make it more robust, I would like to generate search windows with tons of overlap; similar to the idea of convolutional NN. Also, when processing frames, I could run some prediction algorithm to differ whether the vehicle location will be closer or farther from my car. On a smarter note, that will eliminate large redundancy of drawing boxes where cars are not present.


