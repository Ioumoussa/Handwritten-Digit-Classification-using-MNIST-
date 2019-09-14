import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras import backend as K
from keras.optimizers import SGD
from keras.models import load_model


# Loading our Data 

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# examine the size & image dims ( just for checking to avoid any errors of executing & loading )
# printing the number of samples in x_train, x_test , y_train, y_test
print("Initial shape  or dims of x_train", str(x_train.shape))
print("Number of samples in our training data: "+ str(len(x_train)))
print("Number of labels in our training data: "+ str(len(y_train)))
print("Number of samples in our test data: "+ str(len(x_test)))
print("Number of labels in our test data: "+ str(len(y_test)))
print()
print("Dimensions of x_train: " + str(x_train[0].shape))
print("labels in x_train :"+ str(y_train.shape))
print()
print("Dimensions of x_test: " + str(x_test[0].shape))
print("labels in y_train :"+ str(y_test.shape))


#Display some images in our Data Using Opencv

for i in range(0,5):
    rand = np.random.randint(0, len(x_train))
    image = x_train[rand]
    image_name = 'Random sample '+ str(i)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()

#Display some images in our Data Using matpotlib
# Plots 6 images, note subplot's arugments are nrows,ncols,index
# we set the color map to grey since our image dataset is grayscale

plt.subplot(331)
random_num = np.random.randint(0,len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))


plt.subplot(332)
random_num = np.random.randint(0,len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

plt.subplot(333)
random_num = np.random.randint(0,len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

plt.subplot(334)
random_num = np.random.randint(0,len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

plt.subplot(335)
random_num = np.random.randint(0,len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

plt.subplot(336)
random_num = np.random.randint(0,len(x_train))
plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))
 
# Display out plots\n",
plt.show()

# Prepare our dataset for training
# Lets store the number of rows and columns

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]


# Getting our date in the right 'shape' needed for Keras
# We need to add a 4th dimenion to our date thereby changing our
# Our original image shape of (60000,28,28) to (60000,28,28,1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)


# store the shape of a single image 
input_shape = (img_rows, img_cols, 1)

# change our image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255
print('x_train shape:' , x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# one hot coding labels

# Now we one hot encode outputs

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Let's count the number columns in our hot encoded matrix
print ("Number of Classes: " + str(y_test.shape[1]))
num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

# ************************************** Create Our Model ****************************************************
#- We're constructing a simple but effective CNN that uses 32 filters of size 3x3
#- We've added a 2nd CONV layer of 64 filters of the same size 3x2
#- We then downsample our data to 2x2, here he apply a dropout where p is set to 0.25
#- We then flatten our Max Pool output that is connected to a Dense/FC layer that has an output size of 128
#- However we apply a dropout where P is set to 0.5
#- Thus 128 output is connected to another FC/Dense layer that outputs to the 10 categorical units



model = Sequential()
model.add(Conv2D(32,(3,3),                    #(3,3) is the kernel matrix size
                 activation= 'relu',
                 input_shape= input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer= SGD(0.01),
              metrics=['accuracy'])

print(model.summary())


# *************************************** train our model *****************************************************

"""
    "- We place our formatted data as the inputs and set the batch size, number of epochs\n",
    "- We store our model's training results for plotting in future\n",
    "- We then use Kera's molel.evaluate function to output the model's fina performance. Here we are examing Test Loss and Test Accuracy

"""

batch_size = 32
epochs = 1              # to get our model more general to the Data , the epochs should be 25> & <100

history = model.fit(x_train,y_train,
                    batch_size= batch_size,
                    epochs= epochs,
                    verbose=1,
                    validation_data= (x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('test loss : ', score[0])
print('test accuracy ', score[1])


#- Ploting our Loss and Accuracy Charts

history_dict = history.history


# Plotting our loss charts
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)


line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
line2 = plt.plot(epochs, loss_values, label='Training Loss')
plt.setp(line1, linewidth=4.0, marker = '+', markersize=10.0)
plt.setp(line2, linewidth=4.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()



# Plotting our acuracy charts

loss_values = history_dict['acc']
val_loss_values = history_dict['val_acc']
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Accuracy')
line2 = plt.plot(epochs, loss_values, label='Training Accuracy')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=90.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=90.0)
plt.xlabel('Epochs') 
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()


#- Saving the Model, give the path where u wish to save u model

model.save('C:/Users/hp/Desktop/imkab/MY_DeepLearning_Projects/my_mnist_cnn.h5')

#- Load the model

classifier = load_model('C:/Users/hp/Desktop/imkab/MY_DeepLearning_Projects/my_mnist_cnn.h5')



# Now let's test our model we will give some inputs examples & see the results


def Drawing(name, pred, image):
    BLACK = [0,0,0]
    expanded_image= cv2.copyMakeBorder(image,0, 0, 0, imageL.shape[0] ,cv2.BORDER_CONSTANT,value=BLACK)
    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, str(pred), (152, 70) , cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (0,255,210), 2)
    cv2.imshow(name, expanded_image)
    
for i in range(0,7):
    rand= np.random.randint(0, len(x_test))
    image_inputed = x_test[rand]
    
    imageL = cv2.resize(image_inputed, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
    image_inputed = image_inputed.reshape(1,28,28,1)
    
    ## Get Prediction
    res = str(classifier.predict_classes(image_inputed, 1, verbose = 0)[0])
    
    Drawing("Prediction", res, imageL) 
    cv2.waitKey(0)
    
cv2.destroyAllWindows()






