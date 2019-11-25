import keras # 导入Keras
import numpy as np
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras import backend as K
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略avx2警告
BATCH_SIZE = 128  #一个batch由若干条数据构成，每个batch对于网络的一次更新
NUM_CLASSES = 10  #0-9
NUM_EPOCHS = 10	  #一个epoch对应网络的一轮更新

# datapath='D:/python_pro/mldata/mnist.npz'
# data=np.load(datapath)
# x_train=data['x_train']
# y_train=data['y_train']
# x_test=data['x_test']
# y_test=data['y_test']
# download and load the data (split them between train and test sets)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#(image training set,corresponding label training set),(test set,corresponding label test set)

# expand the channel dimension
# input:28*28 gray image
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
#tensorflow (大小,颜色通道数量)
input_shape = (28, 28, 1)

# make the value of pixels from [0, 255] to [0, 1] for further process
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# convert class vectors to binary class matrics
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# define the model
model = Sequential() #类的初始化
model.add(Conv2D(32,(3,3),input_shape=input_shape,activation='relu')) #32 convolution filters of size 3x3 each
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.35))  #防止过拟合,控制需要断开的神经元的比例
model.add(Flatten()) #把多维输入一维化

model.add(Dense(BATCH_SIZE, activation='relu')) #dense代表全连接层,最后接入relu
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax')) #对输入数据的最后一维进行softmax

# define the object function, optimizer and metrics
model.compile(loss=keras.metrics.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
# train
model.fit(x_train, y_train, 
		  batch_size=batch_size, 
		  epochs=epochs,verbose=1, 
		  validation_data=(x_test, y_test))
# evaluate
score_test=model.evaluate(x_test, y_test, verbose=1)
score_train=model.evaluate(x_train, y_train, verbose=1)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (score_train[0],score_train[1]))  #0.0094 0.9974(0.0253 0.9923?)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (score_test[0],score_test[1]))      #0.0228 0.9927
