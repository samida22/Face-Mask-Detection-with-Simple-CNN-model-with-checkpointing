# Face-Mask-Detection-with-Simple-CNN-model-with-checkpointing
Any one interested can use the weights and further explore for better performance.
## Today I am going to try build model for detecting the mask for covid-19 preventions from scratch
### 1. The very first step I did was data pre-processing:
  #### -The dataset consisted of 1376 images, 690 face images with masks and 686 without masks. The original dataset is prepared by Prajna Bhandary and available at Github
	      -import all the necessary packages.
	      -path for our data set is in my drive data folder
	      -we loop throug our data_path and extract target labels for data in each folder
	      -then we convert every images into gray scale resize it and append into dataset an empty folder that I created for storing processed images and store lables into target list with associated images
	      - after thar dataset images are converted into numpy array reshaped and stored into file
	      -same for target lists also also applied to convert into categorical value with the help of np_utils from keras
### 2. now its time for training the datas, I am building every thing from scratch.I will try it later with resnet and vgg19 soon
        from keras.models import Sequential
        from keras.layers import Dense,Activation,Flatten,Dropout
        from keras.layers import Conv2D,MaxPooling2D
        from keras.callbacks import ModelCheckpoint
        now create sequential model and add layers respectively 
        I will do

        Conv2D--MaxPool Conv2D-MaxPool Flatten-Dropout--Dense--Dense

	
           #convolution layer with 200 filters of 3 by 3
	        #relu activation
	        #maxpool layer of 2 by 2 size

	        #convolution layer with 100 filters of 3 by 3
	        #relu activation
	        #maxpool layer of 2 by 2 size

	        #Flattening the layers
 	        #add dense 3 layer 
	        #last two dense layer with 50 and 2 units respectively
	        #last layer will have softmax layer as we have two outputs either with mask or without mask
	


        -now we are going to split data into train test using sklearn.model_selection
        -we are going to keep check points
        - we are doing early stopping in case for worse case. we will be monitoring val_loss as tracking metrics
        -we are going to keep best weights
        -callback_list will be maintained for tracking checkpoints, early stopping points, reduce_learning_rates for better peformance
        -now we are going to compile our model with adam optimizer and metric as accuracy
        -we are going to train for 40 epochs
        -plot of loss vs epochs for training loss and validation loss is shown below
        -next plot of loss vs epochs for training accuracy and validation accuracy
# plots:

        
  
        
