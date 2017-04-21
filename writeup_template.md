# Vehicle Detection Project

# Introduction
In this project, the goal is to write a software pipeline to identify vehicles in a video from a front-facing camera on a car. I tried two different solutions. The first one rely on computer vision and machine learning tecquine while the second one rely on a deep neural network.


# First solution: HOG + SVM

The steps of this project are the following:

* Feature extraction (for each sample of the training set):
  * Perform a Histogram of Oriented Gradients (HOG) feature extraction.
  * Extract binned color features, as well as histograms of color.
  * Concatenate the prevoius results in a vector and normalize
* Train a Linear SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

You can find the code in the IPython notebook named [Vehicle-Detection.ipynb](https://github.com/jokla/CarND-Vehicle-Detection/blob/master/Vehicle-Detection.ipynb).

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## Traning data
I used the data provided by Udacity. Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier. These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

The dataset contains in total 17,760 color images of dimension 64×64 px. 8,792 samples contain a vehicles and 8,968 samples do not. The dataset is then balanced.

## Feature extraction

The code for this step is contained in the first code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

These are the features I used in this project:
* Spatial features: a down sampled copy of the image
* Color histogram features that capture the statistical color information of each image. Cars often have very saturated colors while the background has pale color. This feature could help to indentify the car by the color information.
* Histogram of oriented gradients (HOG): that capture the gradient structure of each image channel and work well under different lighting conditions

As you can see in the next picture, even reducing the size of the image to 32 x 32 pixel resolution, the car itself is still clearly identifiable, and this means that the relevant features are still preserved. This is the function I used to compute the spatial features, it simply resizes the image and flatten to a 1-D vector:

```
# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features
```

The second feature I used are the histograms of pixel intensity (color histograms). The function `color_hist` compute the histogram of the color channels seprately and after concatenates them in a 1-D vector.

```
# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
```

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

I tried various combinations of parameters and these are the paramenters that gaves me the best result:

```
# HOG parameters
orient = 12
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Use all channels

# Spatial size and histogram parameters
spatial_size=(16, 16)
hist_bins=16

```
Here the code that extract the features:
```
### Traning phase
car_features = extract_features(cars, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
print ('Extracting not-car features')
notcar_features = extract_features(notcars, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)


```

As in any machine learning application, we need to normalize our data. In this case I use the function called  StandardScaler() in thePython's sklearn package.

```
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```
Now we can create the labels vector and shuffle and splitting the data into a training and testing set:
```
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

```
We are ready to train our classifier!


### Traning phase

I trained a linear SVM provide by sklearn.svm. 

```
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
```

It takes 26.57 Seconds to train the classifier. I finally got a test accuracy of 99.1%.


### Sliding Window Search

We have to deal now with images coming from a front-facing camera on a car. We need to extract from these full-resolution images some sub-regions and check if they contains a car or not. To extract subregions of the image I used a sliding window approach. It is important to minimize the number of subregions used in order to improve the performane and to avoid looking for cars where we know they cannot be (for example on the sky).

For each subregions we need to compute the feature vector and feed it to the classifier. The classifier, in this case I used a SVM with linear kernel, will predict if there is a car or not in the images.

The function `find_cars` is able to both extract features and make predictions by computing the HOG transform only once for the entire picture. The HOG is then sub-sampled to get all of its overlaying windows. 


Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]


I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps:

![alt text][image5]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


---

### Video Implementation

Finally I tested the pipeline on a video stream. In this case I did not consider each frame individually, in fact we can take advantage of the previous past detections. A deque colletion type is usied to accumulate the detection of the last N frames, in this way is easier to eliminate false positive. The only difference is that the threshold for the heat map will be higher.

This is the result of the detection:

---

# Second solution: SSD (Single Shot MultiBox Detector)

In the last year Convolutional Neural Networks demonstrated to be very succesful for object detection. This is way I was cuouris to test a deep learning approach to detect vehicles. 



# Discussion



