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
 
    x = tflearn.conv_2d(input, 64, 3, activation='relu',regularizer='L2', scope='conv1_1')
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

    x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8',
                                restore=False)

    return x

batchNum=16

Wdirectory="C:/Users/chan/Documents/Visual Studio 2013/Projects/TensorPy/TFlearn/"
model_path = "C:/Users/chan/Desktop/pretrainedModel/"

# the file gen by generated by gen_files_list.py    
files_list = "D:/ML_DATA/TFlearnTest2/TrainingImgList.txt"
from tflearn.data_utils import image_preloader
scaleMap=3
X, Y = image_preloader(files_list, image_shape=(224, 224), mode='file',
                      categorical_labels=True, normalize=False,
                       files_extension=['.jpeg','.jpg', '.png'], filter_channel=True,scaleMapNum=scaleMap)


num_classes = 10 # num of your dataset

# VGG preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(  per_channel=True)  #[ 124.8321991  , 125.27740479 , 123.91309357 , 126.05635834 , 124.04628754,
                                                             # 122.49632263,  126.37927246 , 121.73046875 , 119.99712372 , 125.06362152,
                                                             # 118.81058502 , 117.11448669] 

# VGG Network
x = tflearn.input_data(shape=[None, 224, 224, scaleMap*3], name='input',
                       data_preprocessing=img_prep)

softmax = vgg16(x, num_classes)

SGD=tflearn.optimizers.SGD(learning_rate=0.05,lr_decay=0.96,decay_step=100)
regression = tflearn.regression(softmax, optimizer=SGD,
                                loss='categorical_crossentropy',batch_size=batchNum,validation_batch_size=batchNum,
                                learning_rate=0.05, restore=True)

model = tflearn.DNN(regression, checkpoint_path='C:/Users/chan/Documents/Visual Studio 2013/Projects/TensorPy/myTest/tt',
                    max_checkpoints=3, tensorboard_verbose=0,
                    tensorboard_dir=Wdirectory+"logs")

# Load pre-trained model
#model_file = os.path.join(model_path, "vgg16.tflearn")
#model.load(model_file, weights_only=True)



# Start finetuning
model.fit(testMap, Y, n_epoch=15, validation_set=0.15, shuffle=True,
          show_metric=True, batch_size=batchNum, snapshot_epoch=True,
          snapshot_step=None, run_id='vgg-finetuning')

model.save('C:/Users/chan/Documents/Visual Studio 2013/Projects/TensorPy/myTest/Result/tt')


 