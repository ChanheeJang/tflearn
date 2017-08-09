# -*- coding: utf-8 -*-
'''
Retraining (Finetuning) Example with vgg.tflearn. Using weights from VGG model to retrain
network for a new task (your own dataset).All weights are restored except
last layer (softmax) that will be retrained to match the new task (finetuning).
'''
import matplotlib.pyplot as plt
import os
import numpy as np 
def vgg16(input, num_class):
    if TestMode:
        isRestore=True
    else:
        isRestore=False
    Reg='L2'
    x = tflearn.conv_2d(input, 64, 3, activation='relu',regularizer=Reg, scope='conv1_1',restore=isRestore)
    #x = tflearn.batch_normalization(x,restore=False, name='BatchNorm1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', regularizer=Reg,scope='conv1_2',restore=isRestore)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu',regularizer=Reg, scope='conv2_1')
    #x = tflearn.batch_normalization(x,restore=False, name='BatchNorm2')
    x = tflearn.conv_2d(x, 128, 3, activation='relu',regularizer=Reg, scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu',regularizer=Reg, scope='conv3_1')
    #x = tflearn.batch_normalization(x,restore=False, name='BatchNorm3')
    x = tflearn.conv_2d(x, 256, 3, activation='relu',regularizer=Reg, scope='conv3_2')
    #x = tflearn.batch_normalization(x,restore=False, name='BatchNorm4')
    x = tflearn.conv_2d(x, 256, 3, activation='relu',regularizer=Reg, scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu',regularizer=Reg, scope='conv4_1')
    #x = tflearn.batch_normalization(x,restore=False, name='BatchNorm5')
    x = tflearn.conv_2d(x, 512, 3, activation='relu',regularizer=Reg, scope='conv4_2')
    #x = tflearn.batch_normalization(x,restore=False, name='BatchNorm6')
    x = tflearn.conv_2d(x, 512, 3, activation='relu',regularizer=Reg, scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu',regularizer=Reg, scope='conv5_1')
    #x = tflearn.batch_normalization(x,restore=False, name='BatchNorm7')
    x = tflearn.conv_2d(x, 512, 3, activation='relu',regularizer=Reg, scope='conv5_2')
    #x = tflearn.batch_normalization(x,restore=False, name='BatchNorm8')
    x = tflearn.conv_2d(x, 512, 3, activation='relu',regularizer=Reg, scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8',
                                restore=isRestore)

    return x
##
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, saveDir=os.getcwd(), saveName=""):
    import itertools
    from sklearn.metrics import confusion_matrix
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
 
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(saveDir+'ConfusionMatrix_'+saveName+'.png')
##
def TestAnalysis():
    import cv2
    resultImgDir=saveMismatchDir+"misMatch_"+ModelName+"/"
    if not os.path.exists(resultImgDir):
        os.makedirs(resultImgDir)
        os.makedirs(resultImgDir+"F.Recall/")
        os.makedirs(resultImgDir+"F.Precision/")
        for i in range(len(Label)):
            os.makedirs(resultImgDir+"F.Recall/"+str(i)+"."+Label[i]+"/")
            os.makedirs(resultImgDir+"F.Precision/"+str(i)+"."+Label[i]+"/")

    classNum = len(Y[0])
    ## P&R
    import collections
    counter=collections.Counter(Y.array)
    relevantElements=np.array(list(counter.values()))
    selectedElements=np.zeros(classNum)
    truePositive=np.zeros(classNum)

    # Confusion Matrix
    confusion = np.zeros((classNum,classNum),dtype=int)  # confusion[True][Predict]
   
    count=0
    mismatched=0
    for k in range(0,len(X)):
        predictedLabel = np.argmax(np.squeeze(model.predict([X[k]])) ,0) 
        trueLabel = np.argmax(Y[k],0)

        ## P&R
        selectedElements[predictedLabel]+=1

        ## Img  
        if isSaveMismatch:
            img = cv2.cvtColor((X[k][:,:,0:3]).astype(np.uint8),cv2.COLOR_RGB2BGR)     
            cv2.putText(img,"(True): "+Label[trueLabel],(10,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
     
        if predictedLabel == trueLabel:
            truePositive[predictedLabel]+=1     ## P&R
            confusion[trueLabel][predictedLabel]+=1
            count+=1
        else:
            confusion[trueLabel][predictedLabel]+=1
            if isSaveMismatch:
                cv2.putText(img,Label[predictedLabel] ,(X[0].shape[0]-50,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255)) # put mispredicted label
           
                tt=np.squeeze(model.predict([X[k]]))
                for i in range(3): # print Top-3 prediction
                    maxIndex=np.argmax(tt,0)
                    percent=tt[maxIndex]
                    tt[maxIndex]=0
                    cv2.putText(img, str(maxIndex) +"."+Label[maxIndex] ,(10,X[0].shape[1]-30+i*10), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,0,0)) # put mispredicted label
                    cv2.putText(img, str(percent)                       ,(110,X[0].shape[1]-30+i*10), cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(0,0,0)) # put mispredicted label

                fileName = X.array[k].split('/', 50)[-1]
                fileName = fileName.split('.',50)[0]
                cv2.imwrite(resultImgDir+"F.Recall/"+str(trueLabel)+"."+Label[trueLabel]+"/"+fileName+"_misMatched.jpg",img)
                cv2.imwrite(resultImgDir+"F.Precision/"+str(predictedLabel)+"."+Label[predictedLabel]+"/"+fileName+"_misMatched.jpg",img)

                mismatched+=1
        if isSaveMismatch:
            print("Saving mismatched images.......  Good: "+str(count)+"|| Mismatch: "+ str(mismatched)+" in Total :"+str(len(X)))
  
    ## P&R          
    Recall =     np.round(np.divide(truePositive,relevantElements),4)
    Precision =  np.round(np.divide(truePositive,selectedElements),4)
    Accuracy = round((count)/len(X)*100,4)  
    Fscore = 2*(Recall*Precision)/(Recall+Precision) 

    if isPrecisionRecall:
        whatIsRecall =    "A 클래스의 전체 이미지 중 실제로 맞춘 이미지의 비율 (Recall = A라고 맞춘 이미지 / A 클래스 전체 이미지)"
        whatIsPrecision = "A 라고 예측한 이미지 중 실제 A인 이미지의 비율   (Precision = A라고 맞춘 이미지 / A라고 예측한 이미지)"
        print("Calculating Precision & Recall ......")
        text_file = open(resultImgDir+"/TestResult_"+ModelName+".csv", "w")

        text_file.write("Class,")
        for i in range(classNum):
            text_file.write(Label[i]+",")
 
             
        text_file.write("\n")
        text_file.write("Recall,")
        for i in range(classNum):
            text_file.write(str(Recall[i])+",")

        text_file.write("\n")
        text_file.write("Precision,")
        for i in range(classNum):
            text_file.write(str(Precision[i])+",")

        text_file.write("\n")
        text_file.write("F-Score,")
        for i in range(classNum):
            text_file.write(str(Fscore[i])+",")

        text_file.write("\n")
        text_file.write("# of Imgs,")
        for i in range(classNum):
            text_file.write(str(relevantElements[i])+",")

        text_file.write("\n\nRunID,," + runID+"\n")
        text_file.write("Accuracy,," + str(Accuracy)+"\n")
        text_file.write("Lowest Recall,," + str(Label[np.argmin(Recall)])+","+str(Recall[np.argmin(Recall)])+", Recall :"+whatIsRecall+"\n")
        text_file.write("Lowest Precision,," + str(Label[np.argmin(Precision)])+","+str(Precision[np.argmin(Precision)])+", Precision :"+whatIsPrecision+"\n\n")

        #Confusion Matrix
        text_file.write("Confusion Mat.,")
        for i in range(classNum):
            text_file.write(Label[i]+",")
        for i in range(classNum):
            text_file.write("\n")
            text_file.write(Label[i]+",")
            for j in range(classNum):
                 text_file.write(str(confusion[i][j])+",")

        text_file.close()
        print("Following file has been created :")
        print(resultImgDir+"TestResult_"+ModelName+".csv\n")
    print("Test Accuracy : " + str(Accuracy))
    return confusion

##
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing

DBdirectory = "C:/Users/ATI/Documents/ChanheeJean/vidiDB/MoreData2/"
ModelName = "MoreData2(3scale)"
####### Test Mode ########
TestMode = True
isSaveMismatch=True
isPrecisionRecall=True

testFiles_list =  DBdirectory+"TestImgList.txt" 
saveMismatchDir =  DBdirectory
TestModelPath = 'C:/Users/ATI/Documents/Visual Studio 2013/Projects/TensorPy/tfLearn/vgg-finetuning/FinalModel/'+ModelName
Label=['AOI','xMark','missingSR','scratch','short','pinhole','masking','particle','blur','discolor','crack','good']

##########################

####### Training Mode ########
trainingFiles_list =  DBdirectory+"TrainingImgList.txt" 

finalModelPath = 'C:/Users/ATI/Documents/Visual Studio 2013/Projects/TensorPy/tfLearn/vgg-finetuning/FinalModel/'+ModelName
checkpointPath= 'C:/Users/ATI/Documents/Visual Studio 2013/Projects/TensorPy/tfLearn/vgg-finetuning/'+ModelName
tensorboardPath = 'C:/Users/ATI/Documents/Visual Studio 2013/Projects/TensorPy/tfLearn/logs'

pretrainedModel = "C:/Users/ATI/Documents/ChanheeJean/vidiDB/pretrainedModel/vgg16.tflearn"
runID = 'vgg-' +ModelName
imgSize = (224,224)
batchNum=64
scaleMap=[100,80,60,45,30]#[100,80,60,45,30]
##########################

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
img_prep.add_featurewise_zero_center(mean=[ 103.06277477 , 120.79076593,  101.59317651  ,108.05693601,  126.41018044,
                   107.02081632,  114.68764277,  132.75292483 , 113.27404936,  121.24470792,
                   138.84813068 , 119.18205098 , 127.72194856,  145.45211363  ,125.65270177],  per_channel=True)  

# moreData_Reg(1scale) : mean=[ 103.13388824 , 120.85018158 , 101.65916443]
# moreData_Reg(100-70-40sclae): mean=[ 103.13420982,  120.84993517,  101.65889782 , 111.14320464 , 129.45885459,
  #110.04036431,  123.75994285,  141.37358798  ,121.67629442] 

# moreData_batch(1scale) : mean = [ 103.13388824 , 120.85018158 ,101.65916443]

# MoreData(1scale): mean=[100.78845978, 123.16420746, 103.32084656]
# MoreData2(1scale): mean=[100.06273651, 120.79055786, 101.59283447]
# MoreData(3sclae): mean=[100.78860723 , 123.16408583  ,103.32078441, 106.14882834 ,129.98860877z,
#                 109.78620441  ,111.78917529 , 135.90339299 , 115.39193796,  116.14277245,
#                 140.25949531 , 119.3330677 ,  120.58714975 , 145.25916486,  123.70046009]  
# MoreData2(3scale): mean=[ 103.06277477 , 120.79076593,  101.59317651  ,108.05693601,  126.41018044,
#                   107.02081632,  114.68764277,  132.75292483 , 113.27404936,  121.24470792,
#                   138.84813068 , 119.18205098 , 127.72194856,  145.45211363  ,125.65270177]
 #scaleMap : mean=[ 73.50690228,  89.05697014,  71.33842784 , 74.8667065 ,  90.21627509,   72.29435977 , 78.37673761 , 93.41886061 , 75.08530228]

# data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_crop(imgSize,12) # pad=12
img_aug.add_random_90degrees_rotation([0,1,2,3])
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()

# VGG Network
x = tflearn.input_data(shape=[None, imgSize[0], imgSize[1],dstChannel], name='input',
                       data_preprocessing=img_prep, data_augmentation=img_aug)

softmax = vgg16(x,  len(Y[0]) )  # len(Y[0]) : Number of classes

SGD=tflearn.optimizers.SGD(learning_rate=0.05,lr_decay=0.92,decay_step=100)
regression = tflearn.regression(softmax, optimizer=SGD,
                                loss='categorical_crossentropy',batch_size=batchNum,validation_batch_size=batchNum,
                                learning_rate=0.05, restore=False)

if TestMode:
    model = tflearn.DNN(softmax, max_checkpoints=5, tensorboard_verbose=0)
    # Load Final model
    model.load(TestModelPath, weights_only=True)
 
    # Start Evaluating
    print("Evaluating the TestSet .......")
    confusionMatrix = TestAnalysis()
    resultImgDir=saveMismatchDir+"misMatch_"+ModelName+"/"
    plot_confusion_matrix(confusionMatrix,Label,title = ModelName, saveDir = resultImgDir, saveName=ModelName)
 

else: #TrainingMode
    model = tflearn.DNN(regression, checkpoint_path=checkpointPath,
                    max_checkpoints=5, tensorboard_verbose=0,
                    tensorboard_dir=tensorboardPath)
    # Load pre-trained model
    model.load(pretrainedModel,weights_only=True)
 
    # Start finetuning
    model.fit(X, Y, n_epoch=30, validation_set=0.15, shuffle=True,
              show_metric=True, batch_size=batchNum, snapshot_epoch=True,
              snapshot_step=None, run_id=runID)

    model.save(finalModelPath)



