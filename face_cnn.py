import pandas as pd
from PIL import Image

imgSize = 48 ** 2

#df = pd.read_csv("fer2013/fer2013.csv", sep=",")
imgStr = df['pixels'][0]

imgarr= map(int,imgStr.split(" "))

img = Image.frombytes("L" , (48 , 48) , imgStr , 'raw')

img.show()
from __future__ import division
import pandas as pd
import h5py
import keras
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten , Activation , Convolution2D , MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import os
from tqdm import tqdm
 #  (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
os.chdir('/Users/rishab/projects/retina');

imgSize = 48 ** 2
p = 0
# Reading the Dataset

df = pd.read_csv("fer2013/fer2013.csv", sep=",")

imgStr = df['pixels'][p]

imgarr= map(int,imgStr.split(" "))

# Reconstructing the Image

img = Image.frombytes("L" , (48 , 48) , imgStr , 'raw')

#img.show()

# Saving The Images 
# =============================================================================
# for i in range(df.shape[0]):
#     imgStr = df['pixels'][i]
#     img = Image.frombytes("L" , (48 , 48) , imgStr , 'raw')
#     img.save('images/{}.jpg'.format(i))
#     
# =============================================================================
    
# Argument Parser

parser = argparse.ArgumentParser()

parser.add_argument('-epochs' , type=int , default = 10)
parser.add_argument('-batch_size' , type=int , default = 50)
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size

imgh = 48
imgw = 48

files = os.listdir('images')
print("Number of Images " + str(len(files)))

X = []
y = []

print("Processing Dataset")
for i in tqdm(range(df.shape[0])):
	imgStr = df['pixels'][i]
	imgArr = map(int , imgStr.split(" "))
	imgArr = np.array(imgArr)
	imgArr = np.reshape(imgArr , (48 , 48))
	imgArr = imgArr / 255
	imgArr = imgArr.transpose()
	X.append(imgArr)

X = np.array(X)
y = df['emotion']
y = y.as_matrix()	
y = np.eye(len(np.unique(y)))[y]

input_shape = X.shape

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 3)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Convolution Neural Network

model = Sequential()
#model.add(Conv2D(32 , (3 , 3 ) , input_shape = (48 , 48 , 1)))
model.add(Convolution2D(64, 5, 5, border_mode='valid',
                            input_shape=(48, 48,1)))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))
      
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')) 
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf')) 
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
     
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
     
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))
     
model.add(Flatten())
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.2))
     
model.add(Dense(7))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

# Model Training

model.fit(X_train , y_train , batch_size = batch_size , epochs = epochs , verbose = 2 , validation_data = (X_test , y_test))

# serialize model to JSON
model_json = model.to_json()
with open("fd2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("fd2.h5")
print("Saved model to disk")
