#**Traffic Sign Recognition** 

##Writeup

##MJ Miranda

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./training_class_dist.png "Training Class Dist"
[image10]: ./validation_class_dist.png "Validation Class Dist"
[image11]: ./traffic_sign_viz_1.png "Sample Traffic Sign Viz"
[image12]: ./pre_bw.png "Pre Blackwhite conversion"
[image13]: ./post_bw.png "Post Blackwhite conversion"
[image14]: ./post_bw_norm.png "Post Blackwhite Norm"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mjmiranda-dhi/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic Summary of Data

The code for this step is contained in the second cell of the IPython notebook.

I used the pandas library as well as plain python to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory Visualization of the Dataset

The code for this step is contained in cells 5 - 10 of the IPython notebook.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Training class distribution][image9]   
![Validation class distribution][image10]

Another exploratory visualization is viewing the first three examples of each image, along with class ID and sign name:   

![Sample image visualization][image11]   


### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 11-17th code cells of the IPython notebook.

I decided to convert the images to grayscale because I could not get the network to train properly on 3 channel RGB images. I felt that simplifying the images with grayscale (perceived grayscale luminance based on http://alienryderflex.com/hsp.html) would allow the network to identify shapes/lines better without the added complexities of 3 channels of color. 

Here is an example of a traffic sign image before:


![Pre BW][image12]   

and after grayscaling:

![Post BW][image13]  

As a last step, I normalized the image data because it moved all values between -1 and 1. Giving the inputs equal variance and zero mean makes a well conditioned problem, and the optimizer does not have to go very far to do its job. This supposedly makes it easier for the network to learn the images since the pixel values (0-255) aren't as far off as targets.

![Post BW Normalization][image14]   
  
After converting to grayscale and normalizing, an additional channel was added back into the images.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The training code starts with cell 18 in the IPython notebook. 

The data was pre-split into training, validation and test sets:
* The size of training set is 34799  
* The size of the validation set is 4410   
* The size of test set is 12630  

The training data was shuffled prior to training.


I attempted to Gaussian blur the images, but did not get good results with the learning. I kept minimum kernal to (1,1), but the network would have trouble and the validation accuracy would start to bounce wildly. I decided against the blurring.

With the additional images generated from the blur, I originally attempted to augment the dataset by increasing the number of examples for classes with less than 400 total examples. This required many more epochs, but still caused the learning to either get caught up in local minima or bounce back and forth between weights. I also used several learning rates to settle down the bouncing of the error, but again, ultimately decided to go for a simpler dataset.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for the architecture of my network is in 20th cell of the IPython notebook.
The architecture is similar to the LeNet architecture, which I used as a starting point. I varied between complexity and size. The main changes were starting off with a single channel rather than 3, and ultimately going for deeper layers.

The architecture that ultimately worked was much simpler than the other networks I tried.   
   
Convolutional layer 1:     
* Input:  32x32x1   
* Filter: 5x5x1   
* Output: 28x28x12   
* Activation: ReLu   

Pooling:   
* Input: 28x28x12    
* Filter: 2x2    
* Output: 14x14x12    
* Padding: VALID   
* k: 2,2   
* strides: 2,2   
   
Convolutional layer 2:   
* Input:  14x14x12   
* Filter: 5x5x12   
* Output: 10x10x24   
* Activation: ReLu   
   
Pooling:   
* Input:  10x10x24    
* Filter: 2x2    
* Output: 5x5x24   
* Padding: VALID  
* k: 2,2  
* strides: 2,2   
    
Transitional Layer: Flattening   
* Input: 5x5x24   
* Output: 600  
    
Fully Connected layer 3:    
* Input:  600    
* Output: 240    
* Activation: ReLu   
    
Fully Connected layer 4:   
* Input:  240    
* Output: 120    
* Activation: ReLu       
      
Fully Connected: final layer 5:   
* Input:  120   
* Output: 43 (logits)   


The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 