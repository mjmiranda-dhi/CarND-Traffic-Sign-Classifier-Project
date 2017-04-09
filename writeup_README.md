# **Traffic Sign Recognition** 

## Writeup

## MJ Miranda

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
[image15]: ./germansigns/german_30.jpg "Speed limit 30"
[image16]: ./germansigns/german_exclamation.jpg "General Caution"
[image17]: ./germansigns/german_right.jpg "Right turn"
[image18]: ./germansigns/german_straight.jpg  "Straight"
[image19]: ./germansigns/german_trafficlight.jpg "Traffic light"


## Rubric Points
#### Consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

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

The code for this step is contained in cells 3 - 5 of the IPython notebook.

Here is an exploratory visualization of the data set. It is a bar chart showing how the classes are distributed:

![Training class distribution][image9]   
![Validation class distribution][image10]    
    
Another exploratory visualization by viewing the first three examples of each image, along with class ID and sign name:   

![Sample image visualization][image11]   


### Design and Test a Model Architecture

#### 1. Preprocessing image data

The code for this step is contained in the code cells 6-11 of the IPython notebook.

I decided to convert the images to grayscale because I could not get the network to train properly on 3 channel RGB images. I felt that simplifying the images with grayscale (perceived grayscale luminance based on http://alienryderflex.com/hsp.html) would allow the network to identify shapes/lines better without the added complexities of 3 channels of color. 

Here is an example of a traffic sign image before:


![Pre BW][image12]   

and after grayscaling:

![Post BW][image13]  

As a last step, I normalized the image data because it moved all values between -1 and 1. Giving the inputs equal variance and zero mean makes a well conditioned problem, and the optimizer does not have to go very far to do its job. This supposedly makes it easier for the network to learn the images since the pixel values (0-255) aren't as far off as targets.

![Post BW Normalization][image14]   
  
After converting to grayscale and normalizing, an additional channel was added back into the images.

#### 2. Training, Validation, and Testing data setup

The data setup is contained in the same cells as above, 1-11 in the IPython Notebook.

The data was pre-split into training, validation and test sets:
* The size of training set is 34799  
* The size of the validation set is 4410   
* The size of test set is 12630  

The training data was shuffled prior to training.

I attempted to Gaussian blur the images, but did not get good results with the learning. I kept minimum kernal to (1,1), but the network would have trouble and the validation accuracy would start to bounce wildly. After adjusting the network architecture, I still felt that it was not a good match between the network layout and data, so I decided against the blurring.

With the additional images generated from the blur, I originally attempted to augment the dataset by increasing the number of examples for classes with less than 400 total examples. This required many more epochs, but still caused the learning to either get caught up in local minima or bounce back and forth between weights. I also used several learning rates to settle down the bouncing of the error, but again, ultimately decided to go for a simpler dataset.

 
#### 3. Model Architecture

The code for the architecture of my network is in 13th cell of the IPython notebook.  

The architecture consists of a LeNet model architecture, which I used as a starting point. I varied between complexity and size. The main changes were starting off with a single channel rather than 3, and ultimately going for deeper layers.

The architecture that ultimately worked was much simpler than the other architectures I tried. I opted for deeper layers, and slider wider outputs.  

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale normalized image   			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24  					|
| Flatten   			| outputs 600									|
| Fully connected		| outputs 240									|
| RELU					|												|
| Fully connected		| outputs 120									|
| RELU					|												|
| Fully connected		| outputs 43 (logits one per class)				|
| Softmax				| input: logits, one hot encoding y				|
| Loss					| Reduce mean									|
| Optimizer				| AdamOptimizer									|
|						|												|
 

#### 4. Training the model

The code for training the model is located in the cells 14-19 of the ipython notebook. 

To train the model, I used an epoch of 30 and batch size of 128. I settled on a learning rate of 0.0005, but tried between 0.0001 and 0.0005. With the architecture, 30 epochs and learning rate 0.0005 allowed the network to reach the 0.93 goal on the validation set. With heavier architectures (I tried 3 convolution layers, with 4 fully connected layers, each with deeper filters), and more data (augmented dataset with blurred images for all classes with less than 400 examples), it required many more epochs to complete the task (~100 epochs).

The optimizer used is the AdamOptimizer, a stochastic optimizer, which users the Adam algorithm (https://arxiv.org/pdf/1412.6980v8.pdf) to control the learning rate. 

#### 5. Training process  

The code for calculating the accuracy of the model is located in the 19th cell of the Ipython notebook.

My final model results were:   
* training set accuracy of 0.998    
* validation set accuracy of 0.933   
* test set accuracy of 0.921   

An iterative approach to training was used.
- The first approach was the same LeNet model used in the previous project, with the untouched German sign dataset. This could achieve about 0.85-0.88 accuracy.
- The second approach was to change the LeNet model to add an extra convolution/pooling layer, and an extra fully connected layer, with the untouched dataset. This achieved anywhere between 0.70 - 0.89 accuracy.
- With both approaches, I tried multiple learning rates, and multipe epochs. In most cases, the accuracy would start getting erratic near the ends of the runs.
- The third approach was to augment the data to even out the numbers of examples to get them closer to even. For all classes with less than 400 examples, I doubled the example size by adding a set of Gaussian blurred images with a random kernal size between 1 and 5. This ultimately performed poorly, but when kernal was limited to 1, it performed slightly better.
- The fourth approach was to go back to the original LeNet architecture and try grayscale versions of input. The images were converted to grayscale using the perceived grayscale luminance conversion, and a single channel re-added back into the images prior to processing. This performed well, reaching as high as 0.91 in 30 epochs. 
- The fifth approach was to normalize the grayscale images, with the standard LeNet model. This change helped the network reach the 0.93 accuracy level, but never consistently.
- The sixth approach was to modify the LeNet model to have deeper layers across all layers. This final change helped the network reach 0.93 consistently, and within 40 epochs.
- In most of the cases, the network was overfitting: the training accuracy would tend to 1.00, while the validation accuracy was much lower. In cases like that, increasing the number of observations or decreasing network complexity might work.
- In all cases, epoch and learning rate were adjusted to make sure that the architecture and/or data augmentations weren't fitting well.
- The LeNet model is relevant to the traffic sign application because of all the complexities involved in being able to identify all the different classes and variations of traffic signs.
- The accuracy levels on unseen data (eg test data) can prove that the network can generalize well. The validation accuracy being lower than training accuracy shows that maybe the network is memorizing the training data.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Speed Limit 30][image15] ![General Caution][image16] ![Right Turn][image17] 
![Ahead Only][image18] ![Traffic Signals][image19]

The images might be difficult to classify because of its size within the 32x32 frame, the background, and in the case of the last image, a drawn rendition of the street sign.

Difficulties:
- The ahead only sign may be difficult because of the size of the sign, as well as the green background with the blue sign (when grayscale, the colors may blend)
- The speed limit 30 may be difficult because of the size of the sign within the 32x32 frame
- The general caution sign might have issues because in grayscale, it loses its lower border
- The right turn sign because the angle makes it more of an oblong shape, rather than a circle, as well as background.
- The traffic light sign may be difficult because its a near perfect drawing, and not an actual sign.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in cells 22-26 of the Ipython notebook.

predicted labels [35  1 18 33 18]
actual labels [35, 1, 18, 33, 26]
Accuracy: 80.0%

The accuracy on the larger test set (with the test set images from the original dataset) was 0.92. I think if there were more images, this network would have performed closer to test set accuracy.

It's an interesting misclassification, as the selection between Traffic Signals and General Caution could be easily mistaken: A red outlined triangle, with a single line shape in the middle. In this case, the network could not tell the difference between an exclamation point and the traffic light symbol.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead Only     		| Ahead Only 									| 
| Speed Limit 30		| Speed Limit 30								|
| General Caution		| General Caution								|
| Turn Right Ahead 		| Turn Right Ahead				 				|
| Traffic Signals		| General Caution  								|


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 27th cell of the Ipython notebook.

[ 68.92692566  19.62192917  17.92712021  16.29876328  14.48528671]
[35 36 25 34  9]
correct label: 35

[ 75.30936432  30.30484009  28.09903717  14.01213074   6.99199152]
[1 0 2 4 5]
correct label: 1

[ 80.12018585  35.97002411  32.28469849  28.91520309  16.46436501]
[18 27 26 11 24]
correct label: 18

[ 49.13742065  30.45340157  20.36469269  14.89333248   7.60460186]
[33 35 39 37  1]
correct label: 33

[ 58.02287674  34.46298218  14.70181942  13.72396946  13.52715206]
[18 26 27 24 38]
correct label: 26


![Ahead Only][image18]      
For the first image, the model is relatively sure that this is an ahead only sign (probability of 0.68), and the image does contain an ahead only sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .68         			| Straight Ahead  								| 
| .19     				| Go Straight or Right							|
| .17					| Road Work										|
| .16	      			| Turn Left Ahead					 			|
| .14				    | No Passing     								|

    
![Speed Limit 30][image15]    
For the second image, the model is relatively sure that this is a speed limit30 sign (probability of 0.75), and the image does contain a speed limit 30 sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .75         			| Speed Limit 30 								| 
| .30     				| Speed Limit 20								|
| .28					| Speed Limit 50								|
| .14	      			| Speed Limit 70					 			|
| .06				    | Speed Limit 80 								|

    
![General Caution][image16]    
For the third image, the model is relatively sure that this is a General Caution sign (probability of 0.80), and the image does contain a General Caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .80         			| General Caution 								| 
| .35     				| Pedestrians									|
| .32					| Traffic Signals								|
| .28	      			| Right of way at next intersection				|
| .16				    | Road narrows on the right 					|


![Right Turn][image17]    
For the fourth image, the model is relatively sure that this is a General Caution sign (probability of 0.80), and the image does contain a General Caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .49         			| Turn Right Ahead 								| 
| .30     				| Ahead Only									|
| .30					| Keep left										|
| .14	      			| Go straight or left							|
| .07				    | Speed limit 30 								|



![Traffic Light][image19]    
For the fifth image, the model is relatively sure that this is a General Caution sign (probability of 0.58), but the image does not contain a General Caution sign, it contains a traffic signals sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .58         			| General Caution								| 
| .34     				| Traffic Signals								|
| .14					| Pedestrians									|
| .13	      			| Road narrows on the right						|
| .13				    | Keep right 									|
