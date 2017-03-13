
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.    

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 104 through 132 of the file called `search_classifier.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  

I then explored different color spaces such as RGB, LUV, and YCrCb; as well as different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed first images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Since the figures are pretty small, the range of the parameters is quite narrow. I did not see much of the difference between orient = 6 or 9, same for cells_per_block = 2 or 4. I did not see much difference between the three color spaces that I tried either.

Here is an example using the `YCrCb`,'RGB', and 'LUV' color space and HOG parameters of `orient=6` or `orient=9`, and `cells_per_block=2` or `cells_per_block=4`, not sure whether you can find some difference:


<img src="https://cloud.githubusercontent.com/assets/19335028/23841538/47dcc1f0-076b-11e7-9319-3055c03f1210.png" width="23%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23841556/67b9c7de-076b-11e7-8294-ff51b9d4b842.png" width="23%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23841587/88b09d50-076b-11e7-9c7e-7673c33bd2c6.png" width="23%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23841622/ce2849d2-076b-11e7-9efd-815b9024cff6.png" width="23%"></img> 

<img src="https://cloud.githubusercontent.com/assets/19335028/23841645/fd24622a-076b-11e7-93ea-e66074f0370d.png" width="23%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23841656/0b43a0b4-076c-11e7-8191-48b5e46787d1.png" width="23%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23841668/1f673efc-076c-11e7-9141-5d2967ad2006.png" width="23%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23841677/2b9fb654-076c-11e7-955a-f5ceeabccbe1.png" width="23%"></img> 

####2. Explain how you settled on your final choice of HOG parameters.

Just looking at the pictures, it is hard to make a decision. I selected the color space and hog parameters by training the classifier, because it could render a precision number, which is more objective than my eye balls. I used the parameters suggested in the lesson to start the classifier training. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM and SVM with a rbf kernel using sklearn.model_selection.cross_val_score. SVM with kernel was too slow, so I picked linear SVM. I then compared linear SVM and decision tree. Under the same condition, decision tree was slower and had lower score. so the winner is linear SVM. I also tried LUV and YCrCb color space. They had similar performance when hog_channel is 0, but LUV would throw traceback error when hog_channel is 'ALL' because it has negative pixel values. I tried different HOG parameters, and found out orient=9, pix_per_cell = 8, and cell_per_block = 2 could have the highest score at 0.9909, while the 'Feature vector length' = 8460. The code for this step is contained in lines 134 through 249 of the file called `search_classifier.py`. After classifier was trained, I saved the model by pickle module. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search from 400 to 656 on y axis and full range on x axis. In 'findCar.py', I setup different scales. For each scale, I extracted the feature once then applied sliding window process. This strategy was more efficient and saved me a lot of time. Since cars that are more far away would be smaller while cars that are closer would be bigger. I further narrowed the search area according to the scale size. I finaly settled on the choice showed in lines 175 through 195 in file 'findCar.py'.    



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I also wrote a Vehicle class to find car boxes, then turned the boxes in different scales into heatmap. After applied threshold to filter out the false positive boxes, my code seemed to work pretty well.  Here are some example images:

<img src="https://cloud.githubusercontent.com/assets/19335028/23843069/82050af8-0776-11e7-9c3b-7a412a51c25e.png" width="23%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23843074/8b5abf8a-0776-11e7-9a51-b20213b2b05d.png" width="23%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23843107/ad6f96e0-0776-11e7-8222-2269bc01a214.png" width="23%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23843110/b001fa4c-0776-11e7-8ada-1457c0580792.png" width="23%"></img> 

<img src="https://cloud.githubusercontent.com/assets/19335028/23843150/0a194080-0777-11e7-90fc-b03117968390.png" width="23%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23843152/0d294522-0777-11e7-9893-40e3bd03bda8.png" width="23%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23843155/0efed826-0777-11e7-9c61-03491c45cf2d.png" width="23%"></img> <img src="https://cloud.githubusercontent.com/assets/19335028/23843162/11af203a-0777-11e7-9b3c-2bf164821176.png" width="23%"></img> 

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/tOZhbInBmNU)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As showed in file 'video_process.py', I moved heatmap methods into the Vehicle class, so I could find each individual vehicle and append the vehicle heatmap to a class variable called 'recentvehicles'. I recorded the positions of positive detections in each frame of the video. I summed up the heatmaps of current and 5 previous frames. I then applied threshold to the sum of the heatmaps of 6 frames and drew the boxes on the image. I used `scipy.ndimage.measurements.label()` to identify individual blobs (cars) in the heatmap. I constructed bounding boxes to cover the area of each blob detected. Since I used different scales, sometimes smaller boxes can be drawn inside the bigger boxes. To address this problem, I modified the apply_threshold() function. I added one line: 'heatmap[heatmap > threshold] = 255' at 187 in file 'video_process.py', then the inside smaller boxes were eliminated.   

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One big problem of my code is that it takes quite a while to process a video, which is not acceptable for a self-driving car. Another approach would be using Faster-RCNN to detect the vehicles and other object. I ran my code on the 'challenge_video' in advanced_lane_finding project. It did not work very well on the vehicles that are very close. I think it might due to the fact that my classifier was trained on 64x64 images, which did not work well on bigger images. One way I can think of is to crop bigger car images and non-car images, then train another classifier, which would be applied on bigger windows.     

