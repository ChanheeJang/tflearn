# -*- coding: utf-8 -*-
'''
Retraining (Finetuning) Example with vgg.tflearn. Using weights from VGG model to retrain
network for a new task (your own dataset).All weights are restored except
last layer (softmax) that will be retrained to match the new task (finetuning).
'''
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os


def vgg16(input, num_class):

    x = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8')

    return x


#data_dir = "/path/to/your/data"

num_classes = 10 # num of your dataset


# VGG preprocessinga
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(mean=[ 124.84864044 , 125.29920959 , 124.0764389 ],per_channel=True)

batchNum=16

# VGG Network
x = tflearn.input_data(shape=[None, 224, 224, 3], name='input',
                       data_preprocessing=img_prep)
softmax = vgg16(x, num_classes)
SGD = tflearn.optimizers.SGD(learning_rate=0.05,lr_decay=0.92,decay_step=1000)
regression = tflearn.regression(softmax, optimizer=SGD, batch_size=batchNum,validation_batch_size=batchNum,
                                 loss='categorical_crossentropy',
                                learning_rate=0.05, restore=True)
 
model = tflearn.DNN(softmax, checkpoint_path='C:/Users/ATI/Documents/Visual Studio 2013/Projects/TensorPy/tfLearn/vgg-finetuning/',
                    max_checkpoints=3, tensorboard_verbose=0,
                    tensorboard_dir="./logs")





test_model_file = "C:/Users/chan/Documents/Visual Studio 2013/Projects/TensorPy/myTest/YOLOdata-1750"
test_files_list = "D:/ML_DATA/TFlearnTestImg(TestImgs)2/TestImgList.txt"   #mean=[95.76175442, 107.05757606, 88.9937943] 0,1,2(bad,small,good) 



from tflearn.data_utils import image_preloader
test_x, test_y = image_preloader(test_files_list, image_shape=(224, 224),mode='file',
                       categorical_labels=True, normalize=False,  ## categorical_labels must be set to True
                       files_extension=['.jpeg','jpg', '.png'], filter_channel=True)
 


model.load(test_model_file)


# Start Evaluating
score = model.evaluate(test_x,test_y,batch_size=batchNum)
print("Test Accuracy : %0.4f%%" %(score[0]*100))

 
#prediction = model.predict([test_x[0]])
#print("Test Accuracy : %s" %str(prediction[0]))


#import cv2
#import numpy as np 
#count=0
#for k in range(0,len(test_x)):
#    img = (test_x[k]).astype(np.uint8)
#    label=str(np.argmax(test_y[k],0))
 
#    if np.argmax(np.squeeze(model.predict([test_x[k]])) ,0) == np.argmax(test_y[k],0):
#        count+=1
#        cv2.putText(img,"Label: "+str(np.argmax(np.squeeze(model.predict([test_x[k]])) ,0)) ,(test_x[0].shape[0]-80,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
#    else:
#        cv2.putText(img,"Label: "+str(np.argmax(np.squeeze(model.predict([test_x[k]])) ,0)) ,(test_x[0].shape[0]-80,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
#    cv2.putText(img,"Label(T): "+label,(10,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255))
#    cv2.imshow("res",img)
#    c=cv2.waitKey(0) 
#    if c ==13:
#        continue
#    print (model.predict([test_x[k]]))
 
 
#print ("Total : "+str(len(test_x))+" || correct : "+ str(count) + " || Percent : " + str(count/len(test_x)))
 
 
