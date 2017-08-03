# -*- coding: utf-8 -*-
'''
Retraining (Finetuning) Example with vgg.tflearn. Using weights from VGG model to retrain
network for a new task (your own dataset).All weights are restored except
last layer (softmax) that will be retrained to match the new task (finetuning).
'''
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os

saveMismatch=True

files_list =  "C:/Users/ATI/Documents/ChanheeJean/vidiDB/MoreData/TestImgList.txt" 
saveMismatchDir =  "C:/Users/ATI/Documents/ChanheeJean/vidiDB/MoreData/"
#checkpointPath= 'C:/Users/ATI/Documents/Visual Studio 2013/Projects/TensorPy/tfLearn/vgg-finetuning/scaleMap'
#tensorboardPath = 'C:/Users/ATI/Documents/Visual Studio 2013/Projects/TensorPy/tfLearn/logs'
finalModelPath = 'C:/Users/ATI/Documents/Visual Studio 2013/Projects/TensorPy/tfLearn/vgg-finetuning/FinalModel/scaleMap'
batchNum=64
scaleMap=3
imgSize = (224,224)
Label=['Good','Bad']

precalcMean = [ 73.50690228,  89.05697014,  71.33842784 , 74.8667065 ,  90.21627509,  72.29435977 , 78.37673761 , 93.41886061 , 75.08530228]

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

from tflearn.data_utils import image_preloader
test_x, test_y = image_preloader(files_list, image_shape=imgSize,mode='file',
                       categorical_labels=True, normalize=False,  ## categorical_labels must be set to True
                       files_extension=['.jpeg','jpg', '.png'], filter_channel=True,scaleMapNum=scaleMap)
 

# VGG preprocessinga
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(mean=precalcMean,per_channel=True)

# VGG Network
x = tflearn.input_data(shape=[None, imgSize[0],imgSize[1], scaleMap*3], name='input',
                       data_preprocessing=img_prep)
softmax = vgg16(x, len(test_y[0]))
SGD = tflearn.optimizers.SGD(learning_rate=0.05,lr_decay=0.92,decay_step=1000)
regression = tflearn.regression(softmax, optimizer=SGD, batch_size=batchNum,validation_batch_size=batchNum,
                                 loss='categorical_crossentropy',
                                learning_rate=0.05, restore=True)
 
model = tflearn.DNN(softmax, max_checkpoints=3, tensorboard_verbose=0)

model.load(finalModelPath)

# Start Evaluating
score = model.evaluate(test_x,test_y,batch_size=batchNum)
print("Test Accuracy : %0.4f%%" %(score[0]*100))


if saveMismatch:
    import cv2
    import numpy as np 
    resultImgDir=saveMismatchDir+"misMatch/"
    if not os.path.exists(resultImgDir):
        os.makedirs(resultImgDir)
    count=0
    mismatched=1
    for k in range(0,len(test_x)):
        img = cv2.cvtColor((test_x[k][:,:,0:3]).astype(np.uint8),cv2.COLOR_RGB2BGR)
        
        cv2.putText(img,"(True): "+Label[np.argmax(test_y[k],0)],(10,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
        if np.argmax(np.squeeze(model.predict([test_x[k]])) ,0) == np.argmax(test_y[k],0):
            count+=1
            #cv2.putText(img,Label[np.argmax(np.squeeze(model.predict([test_x[k]])) ,0)] ,(test_x[0].shape[0]-80,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
        else:
            cv2.putText(img,Label[np.argmax(np.squeeze(model.predict([test_x[k]])) ,0)] ,(test_x[0].shape[0]-30,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
            fileName = test_x.array[k].split('/', 20)[-1]
            fileName = fileName.split('.',20)[0]
            cv2.imwrite(resultImgDir+fileName+"_misMatched.jpg",img)
            mismatched+=1
        print("Saving mismatched images.......  Good: "+str(count)+"|| Mismatch: "+ str(mismatched)+" in Total :"+str(len(test_x)))

        #print (model.predict([test_x[k]]))
 
 
#print ("Total : "+str(len(test_x))+" || correct : "+ str(count) + " || Percent : " + str(count/len(test_x)))
 
 