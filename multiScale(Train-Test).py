# -*- coding: utf-8 -*-
'''
Retraining (Finetuning) Example with vgg.tflearn. Using weights from VGG model to retrain
network for a new task (your own dataset).All weights are restored except
last layer (softmax) that will be retrained to match the new task (finetuning).
'''
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os

DBdirectory = "C:/Users/ATI/Documents/ChanheeJean/vidiDB/MoreData/"
ModelName = "MoreData(1scale)"
####### Test Mode ########
TestMode = False
saveMismatch=False
testFiles_list =  DBdirectory+"TestImgList.txt" 
saveMismatchDir =  DBdirectory
TestModelPath = 'C:/Users/ATI/Documents/Visual Studio 2013/Projects/TensorPy/tfLearn/vgg-finetuning/FinalModel/'+ModelName
Label=['AOI','xMark','missingSR','scratch','short','pinhole','masking','particle','blur','discolor','crack']

##########################

####### Training Mode ########
trainingFiles_list =  DBdirectory+"TrainingImgList.txt" 

finalModelPath = 'C:/Users/ATI/Documents/Visual Studio 2013/Projects/TensorPy/tfLearn/vgg-finetuning/FinalModel/'+ModelName
checkpointPath= 'C:/Users/ATI/Documents/Visual Studio 2013/Projects/TensorPy/tfLearn/vgg-finetuning/'+ModelName
tensorboardPath = 'C:/Users/ATI/Documents/Visual Studio 2013/Projects/TensorPy/tfLearn/logs'

pretrainedModel = "C:/Users/ATI/Documents/ChanheeJean/vidiDB/pretrainedModel/vgg16.tflearn"
runID = 'vgg-moreData(1Scale)'
imgSize = (224,224)
batchNum=64
scaleMap=1
##########################


def vgg16(input, num_class):
    if TestMode:
        isRestore=True
    else:
        isRestore=False
    x = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1',restore=isRestore)
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2',restore=isRestore)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')
    #x = tflearn.batch_normalization(x)

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')
    #x = tflearn.batch_normalization(x)

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')
    #x = tflearn.batch_normalization(x)

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')
    #x = tflearn.batch_normalization(x)

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8',
                                restore=isRestore)

    return x


if type(scaleMap) is int: 
    dstChannel = scaleMap*3
else:
    dstChannel =  len(scaleMap)*3


if TestMode:
    files_list = testFiles_list
else:
    files_list = trainingFiles_list


from tflearn.data_utils import image_preloader

X, Y = image_preloader(files_list, image_shape=imgSize, mode='file',
                      categorical_labels=True, normalize=False,
                       files_extension=['.jpeg','.jpg', '.png'], filter_channel=True,scaleMapNum=scaleMap)  
 
# VGG preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)  
# MoreData(1scale): mean=[100.78845978, 123.16420746, 103.32084656]
# MoreData: mean=[100.78860723 , 123.16408583  ,103.32078441, 106.14882834 ,129.98860877,
#                 109.78620441  ,111.78917529 , 135.90339299 , 115.39193796,  116.14277245,
#                 140.25949531 , 119.3330677 ,  120.58714975 , 145.25916486,  123.70046009] 
# scaleMap : mean=[ 73.50690228,  89.05697014,  71.33842784 , 74.8667065 ,  90.21627509,   72.29435977 , 78.37673761 , 93.41886061 , 75.08530228]

# data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_crop(imgSize,12) # pad=12
img_aug.add_random_90degrees_rotation([0,1,2,3])
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()

# VGG Network
x = tflearn.input_data(shape=[None, imgSize[0], imgSize[1],dstChannel], name='input',
                       data_preprocessing=img_prep, data_augmentation=img_aug)

softmax = vgg16(x,  len(Y[0]) )

SGD=tflearn.optimizers.SGD(learning_rate=0.05,lr_decay=0.92,decay_step=100)
regression = tflearn.regression(softmax, optimizer=SGD,
                                loss='categorical_crossentropy',batch_size=batchNum,validation_batch_size=batchNum,
                                learning_rate=0.05, restore=False)

if TestMode:
    model = tflearn.DNN(softmax, max_checkpoints=5, tensorboard_verbose=0)
    # Load Final model
    model.load(TestModelPath, weights_only=True)

    # Start Evaluating
    score = model.evaluate(X,Y,batch_size=batchNum)
    print("Test Accuracy : %0.4f%%" %(score[0]*100))

else: #TrainingMode
    model = tflearn.DNN(regression, checkpoint_path=checkpointPath,
                    max_checkpoints=5, tensorboard_verbose=0,
                    tensorboard_dir=tensorboardPath)
    # Load pre-trained model
    model.load(pretrainedModel, weights_only=True)
 
    # Start finetuning
    model.fit(X, Y, n_epoch=60, validation_set=0.15, shuffle=True,
              show_metric=True, batch_size=batchNum, snapshot_epoch=True,
              snapshot_step=None, run_id=runID)

    model.save(finalModelPath)


if TestMode and saveMismatch:
    import cv2
    import numpy as np 
    resultImgDir=saveMismatchDir+"misMatch/"
    if not os.path.exists(resultImgDir):
        os.makedirs(resultImgDir)
    count=0
    mismatched=1
 
    for k in range(0,len(X)):
        img = cv2.cvtColor((X[k][:,:,0:3]).astype(np.uint8),cv2.COLOR_RGB2BGR)
        
        cv2.putText(img,"(True): "+Label[np.argmax(Y[k],0)],(10,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
        if np.argmax(np.squeeze(model.predict([X[k]])) ,0) == np.argmax(Y[k],0):
            count+=1
        else:
            cv2.putText(img,Label[np.argmax(np.squeeze(model.predict([X[k]])) ,0)] ,(X[0].shape[0]-50,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
            fileName = X.array[k].split('/', 50)[-1]
            fileName = fileName.split('.',50)[0]
            cv2.imwrite(resultImgDir+fileName+"_misMatched.jpg",img)
            mismatched+=1
        print("Saving mismatched images.......  Good: "+str(count)+"|| Mismatch: "+ str(mismatched)+" in Total :"+str(len(X)))
