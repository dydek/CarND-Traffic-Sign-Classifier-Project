# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./validation_data/example_00002.png "Traffic Sign 1"
[image5]: ./validation_data/example_00003.png "Traffic Sign 2"
[image6]: ./validation_data/example_00010.png "Traffic Sign 3"
[image7]: ./validation_data/example_00012.png "Traffic Sign 4"
[image8]: ./validation_data/example_00013.png "Traffic Sign 5"
[image9]: ./examples/training_1.png "Trainign 1"
[image10]: ./validation_data/example_00016.png "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dydek/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used standar python functions to calculate these values:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

```python

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_test[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))
```


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because in that case I got only one layer to process instead of 3 ( for RGB images ). The reason is to try to minimalize the errors during the learning process.
For converting I was using this function:

```python
def rgb2gray(rgb):
    return np.sum(rgb / 3, axis=3, keepdims=True)
```

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data. It's important to have for all features one scale. In the current solution 
I've done it using simple function ( based on numpy operation ) : 

```python
def normalize(images):
    # according to https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/91cc6685-08df-4277-b53d-3a792b02420d/concepts/71191606550923
    return (images-125)/125
```

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        | 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1	Greyscale	| 
| Convolution 3x3   | 1x1 stride, padding 'VALID' , Output 28x28x10 |
| RELU					|						|
| Max Pooling	      	| 2x2 stride,  outputs 14x14x10|
| Convolution 3x3   | 1x1 stride, padding 'VALID' , Output 10x10x20 |
| RELU					|						|
| Max Pooling	      	| 2x2 stride,  outputs 5x5x20|
| Flatten           | Output: 500 |
| Fully connected   | Output 250  |
| Relu              |             |
| Fully connected   | Output 125  |
| Relu              |             | 
| Fully connected   | Output 43   |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I was trying a different type of batches, epochs and learning rates. After several tries, I've found that the best results I'm getting for ~ 20 epoch, small batch size ( ~ 64 ) and learning rate 0.0008. For the highest learning rate, the network was achieving the accuracy around 0.93 faster, but it had a problem to go higher. A similar situation happened with a larger batch size. 
That's why I've decided to use a small batch size, small learning rate, and more epochs. According to my test, it gave the best results.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.954
* validation set accuracy of 0.9503
* test set accuracy of 0.9365

This is how trainign process looked like:

![alt text][image9]

I've chosen a classic LeNet architecture. My decision was dictated by my very limited knowledge about neural networks, so 
I've decided to use something which was designed to process that kind of that, and LeNet definitely is. Getting accuracy above 0.94 for my first NN is great, and it's proving that it works ( more or less for the test data ). I believe this is a good first step to optimize it further.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

( I've chosen 6 instead of 5 )

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image10]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 20   		| Speed limit 30	| 
| Keep left    			| Keep left   	|
| Keep left ahead					| Keep left ahead|
| Keep left	      		| Keep left		|
| Yield			| Yield    	|
| No entry      | No entry


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. 
This is the lower result that I've got for test set. Probably the reason is that the images are not cropped properly.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 81st cell of the Ipython notebook.

For the first image I'm getting almost 90% that the sign is 30kmh , and only 0.01 that it's 20kmh, which is super strange for me.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .89        			| 30kmh  		| 
| .9     				| Roundabout mandatory 				|
| .024					| Go straight or left				|
| .02	      			| General coution			|
| .01				    | Speed limit 20     |


For the socond image I'm getting 100% keep left ( which is true ), 
similar situation is happening for the rest ( 3rd - 99.99 Turn left ahead , 4th 100 % keep left, 5th 100% yield, 6th 99.95 % no entry ). 

So the error is only for the first sign. My wild guess why is that is the super small training set of 20kmh signs.