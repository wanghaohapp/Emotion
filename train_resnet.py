#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import ResNet as RN
from tensorflow.python.keras.optimizers import Adam,RMSprop, SGD
from tensorflow.python.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping,ModelCheckpoint,LearningRateScheduler
from tensorflow import keras
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from keras import backend as K
K.clear_session()
import time
import cv2
from mtcnn.mtcnn import MTCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
root_file = '/home/app/data/AffectNet/align_face/image'
#micro训练权重
model_path = '/home/app/program/micro_emotion/resnet10/micro/model.h5'
#macro训练权重,在进行micro表情训练时进行初始化
pre_model = '/home/app/program/micro_emotion/resnet10/macro/model.h5'
#进行macro训练的初始赋值权重
ori_model = '/home/app/program/micro_emotion/resnet10/pre_macro/model.h5'

#直接导入影像数据进行训练,与train函数连用
def load_data(path):
    fopen = open(path)
    path_list = fopen.readlines()
    data = []
    labels = []
    for _file in path_list:
        img_path = _file.split()[0]
        label = _file.split()[1]
        labels.append(eval(label))
        img = cv2.imread(os.path.join(root_file,img_path))
        img = cv2.resize(img,(224,224))
        img = img_to_array(img)
        data.append(img)

    labels = np.array(labels)
    data = np.array(data)
    labels = to_categorical(labels, num_classes=8)

    return data, labels


def train(train_data, train_labels, test_data, test_labels):
    model = RN.resnet(weights=model_path,input_tensor=None,input_shape=(224,224,3),classes=8)
    logging = TensorBoard(log_dir='/home/app/program/micro_emotion')
    lr_decay = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=3)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001,decay=0.001/100),
                  metrics=['accuracy'])
    model.fit(train_data,train_labels,epochs=100,batch_size=32,
              validation_data=(test_data, test_labels),callbacks=[logging,lr_decay])

    model.save(model_path)

#训练macro表情,利用生成器获取数据,并进行训练
def fit_train(train_path,test_path):

    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    print("获取测试数据... ...")
    test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=32)
    print("获取训练数据... ...")
    train_generator = train_datagen.flow_from_directory(train_path,target_size=(224,224),batch_size=32)

    print("开始训练... ...")
    logging = TensorBoard(log_dir='/home/app/program/micro_emotion/log')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    model_check = ModelCheckpoint(filepath=pre_model,monitor='val_loss',save_best_only=True)
    lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=0.001)
    model = RN.resnet10(include_top = False, weights = ori_model, pooling = 'avg', input_shape=(224, 224, 3), classes=7)

    x = model.output
    x = Dense(7, activation='softmax', name='fc8_5')(x)
    model = Model(inputs=model.input, outputs=x)
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01,momentum=0.9,decay=0.1 / 20),
                  metrics=['accuracy'])
    history = model.fit_generator(train_generator,steps_per_epoch=346,epochs=60,
                                  validation_data=test_generator,validation_steps=38,
                                  callbacks=[logging,early_stopping,model_check,lr_decay])
#利用获取的网络进预测
def predict_emotion():
    emotion_list = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    model = RN.resnet10(include_top=False, pooling='avg',input_tensor=None, input_shape=(224, 224, 3),
                              classes=7)
    x = model.output
    x = Dense(7, activation='softmax', name='fc8_5')(x)
    model = Model(inputs=model.input, outputs=x)
    model.load_weights('/home/app/program/micro_emotion/resnet10/macro/model.h5', by_name=True)
    dector = MTCNN()
    # img_path = '/home/app/data/beiyou/basic/Image/test/train/3/test_0063.jpg'
    # img = cv2.imread(img_path)
    # # t = dector.detect_faces(img)
    # # point = t[0]['box']
    # # face = img[point[1]:point[1] + point[3], point[0]:point[0] + point[2]]
    # face = cv2.resize(img, (224, 224))
    # cv2.imshow('face',face)
    # face = img_to_array(face)
    # face = face.reshape((-1, 224, 224, 3))
    # out = model.predict(face)
    # print(emotion_list[out.argmax()])
    # cv2.waitKey()
    capture = cv2.VideoCapture(0)
    while (True):
        ref, frame = capture.read()
        img = frame.copy()
        t = dector.detect_faces(img)

        point = t[0]['box']
        #face = img[point[1]:point[1] + point[3], point[0]:point[0] + point[2]]
        keypoint1 = np.float32([[30, 30], [70, 30], [50, 80]])
        keypoint2 = []
        keypoint2.append(t[0]['keypoints']['left_eye'])
        keypoint2.append(t[0]['keypoints']['right_eye'])
        x = np.array(t[0]['keypoints']['mouth_left'], dtype=np.float32)
        y = np.array(t[0]['keypoints']['mouth_right'], dtype=np.float32)
        center = (x + y) / 2
        keypoint2 = np.array(keypoint2, dtype=np.float32)
        keypoint2 = np.row_stack((keypoint2, center))

        matrix = cv2.getAffineTransform(keypoint2, keypoint1)
        output = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        face = output[:100, :100]

        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = face.reshape((-1, 224, 224, 3))
        start = time.clock()
        out = model.predict(face)
        end = time.clock()
        print('耗时{}s'.format(end - start))
        #print(out)
        print(emotion_list[out.argmax()])
        cv2.rectangle(frame, (point[0], point[1]), (point[0] + point[2], point[1] + point[3]), (0, 255, 0), 2)
        cv2.imshow('1', frame)
        cv2.waitKey(1)
#网络解冻
def unfreeze(model,count):
    for i in range(135):
        if ((model.layers[i].name == 'concatenate_{0}'.format(count)) or (model.layers[i].name == 'average_{0}'.format(count))
                or (model.layers[i].name == 'multiply_{0}'.format(count))
                or (model.layers[i].name == 'add_{0}'.format((2*count + 1)))
                or (model.layers[i].name == 'activation_{0}'.format((2*count + 2)))):
            model.layers[i].trainable = True


    # concat = 'concatenate_{0}'.format(count)
    # ave = 'average_{0}'.format(count)
    # mul = 'multiply_{0}'.format(count)
    # add = 'add_{0}'.format((2*count + 1))
    # act = 'activation_{0}'.format((2*count + 2))
    # model.concat.trainable = True
    # model.ave.trainable = True
    # model.mul.trainable = True
    # model.add.trainable = True
    # model.act.trainable = True
    return model

def frozen(model):
    model.trainable = False
    for i in range(135):
        if ((model.layers[i].name == 'concatenate') or (model.layers[i].name == 'res2a_branch2') or (model.layers[i].name == 'bn2a_branch2')
                or (model.layers[i].name == 'res3a_branch2') or (model.layers[i].name == 'bn3a_branch2') or (model.layers[i].name == 'res4a_branch2')
                or (model.layers[i].name == 'bn4a_branch2') or (model.layers[i].name == 'res5a_branch2') or (model.layers[i].name == 'bn5a_branch2')
                or (model.layers[i].name == 'average') or (model.layers[i].name == 'multiply') or (model.layers[i].name == 'add_1')
                or (model.layers[i].name == 'activation_2')):
            model.layers[i].trainable = True
    # model.concatenate.trainable = True
    #
    # model.res2a_branch2.trainable = True
    # model.bn2a_branch2.trainable = True
    # model.res3a_branch2.trainable = True
    # model.bn3a_branch2.trainable = True
    # model.res4a_branch2.trainable = True
    # model.bn4a_branch2.trainable = True
    # model.res5a_branch2.trainable = True
    # model.bn5a_branch2.trainable = True
    #
    #
    # model.average.trainable = True
    # model.multiply.trainable = True
    # model.add_1.trainable = True
    # model.activation_2.trainable = True
    for count in range(1,8):
        model = unfreeze(model,count)
    return model



#训练micro表情(不需要加attention block)
def finetune(train_path,test_path):
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    print("获取测试数据... ...")
    test_generator = test_datagen.flow_from_directory(test_path, target_size=(224, 224), batch_size=4)
    print("获取训练数据... ...")
    train_generator = train_datagen.flow_from_directory(train_path, target_size=(224, 224), batch_size=4)

    print("开始训练... ...")
    logging = TensorBoard(log_dir='/home/app/program/micro_emotion/log')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    model_check = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
    lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=0.001)
    model = RN.resnet10(include_top=False, pooling='avg',
                        input_shape=(224, 224, 3), classes=5)
    x = model.output
    x = Dense(5, activation='softmax', name='fc8_micro')(x)
    model = Model(inputs=model.input, outputs=x)
    model.load_weights(pre_model,by_name=True)
    #model = frozen(model)
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01, momentum=0.9, decay=0.1 / 10),
                  metrics=['accuracy'])
    history = model.fit_generator(train_generator, steps_per_epoch=62, epochs=30,
                                  validation_data=test_generator, validation_steps=30,
                                  callbacks=[logging, early_stopping, model_check, lr_decay])

#获取混淆矩阵
def confusionmatrix(img_path,dim):
    labels = os.listdir(img_path)
    labels.sort()
    files = []
    for label in labels:
        imgpath = os.path.join(img_path,label)
        imgnames = os.listdir(imgpath)
        for imgname in imgnames:
            s = '{0}/{1} {0}'.format(label,imgname)
            files.append(s)

    np.random.shuffle(files)
    confusion_matrix = np.zeros((dim, dim), np.int)
    confusion_matrix_float = np.zeros((dim, dim), np.float)

    #构建模型
    model = RN.resnet10(include_top=False, pooling='avg', input_tensor=None, input_shape=(224, 224, 3),
                        classes=5)
    x = model.output
    x = Dense(5, activation='softmax', name='fc8_micro')(x)
    model = Model(inputs=model.input, outputs=x)
    model.load_weights(model_path, by_name=True)


    #获取混淆矩阵
    for file in files:
        path = file.split()[0]
        sign = eval(file.split()[1])

        img = cv2.imread(os.path.join(img_path,path))
        face = cv2.resize(img, (224, 224))
        face = img_to_array(face)
        face = face.reshape((-1, 224, 224, 3))
        predict_value = int(model.predict(face).argmax())
        confusion_matrix[sign, predict_value] += 1
    print(confusion_matrix)
    c_sum = confusion_matrix.sum(axis=1)
    confusion_matrix_float = confusion_matrix / c_sum
    print(confusion_matrix_float)



if __name__ == '__main__':
    # test_path = '/home/app/data/beiyou/basic/Image/test/test'
    # train_path = '/home/app/data/beiyou/basic/Image/test/train'
    # fit_train(train_path,test_path)
    # finetune(train_path,test_path)
    predict_emotion()
    # img_path = '/home/app/data/micro_exression/casme2_img/face'
    # confusionmatrix(img_path,5)





