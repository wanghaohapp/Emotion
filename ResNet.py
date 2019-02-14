#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.keras.utils import layer_utils, plot_model
from tensorflow.python.keras.layers import Lambda
#from keras.backend import slice
# from tensorflow.python.keras.layers.merge import Split_Average
import tensorflow as tf
from keras import backend as K
K.clear_session()


def conv_block_10(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
    filter1,filter2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, kernel_size, strides=strides, padding='same', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis,name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    shortcut = Conv2D(filter2,(1,1),strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x,shortcut])
    x = Activation('relu')(x)
    return x

def conv_attention_block_10(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
    filter1, filter2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, kernel_size, strides=strides, padding='same', name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    y1 = x
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    y2 = x

    shortcut = Conv2D(filter2, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    y3 = shortcut

    x = layers.add([x, shortcut])

    y = layers.concatenate([y1,y3,y2],axis=bn_axis)

    y = Conv2D((filter2*3),(1, 1),name=conv_name_base + '2')(y)
    y = BatchNormalization(axis=bn_axis, name=bn_name_base + '2')(y)
    t = Lambda(lambda y: tf.split(y,3,axis=3))(y)

    y = layers.average([t[0],t[1],t[2]])
    y = layers.multiply([x,y])

    x = layers.add([x, y])
    x = Activation('relu')(x)
    return x

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
  """The identity block is the block that has no conv layer at shortcut.

  Arguments:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names

  Returns:
      Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)

  x = Conv2D(
      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
          x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)

  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

  x = layers.add([x, input_tensor])
  x = Activation('relu')(x)
  return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,
                                                                          2)):
  """A block that has a conv layer at shortcut.

  Arguments:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the first conv layer in the block.

  Returns:
      Output tensor for the block.

  Note that from stage 3,
  the first conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well
  """
  filters1, filters2, filters3 = filters
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = Conv2D(
      filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(
          input_tensor)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)

  x = Conv2D(
      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
          x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)

  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

  shortcut = Conv2D(
      filters3, (1, 1), strides=strides, name=conv_name_base + '1')(
          input_tensor)
  shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

  x = layers.add([x, shortcut])
  x = Activation('relu')(x)
  return x


def conv_attention_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(
        filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(
        input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    y1 = x
    x = Activation('relu')(x)

    x = Conv2D(
        filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
        x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    y2 = x
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    y3 = x

    shortcut = Conv2D(
        filters3, (1, 1), strides=strides, name=conv_name_base + '1')(
        input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    y4 = shortcut
    x = layers.add([x, shortcut])
    #将y1,y2,y3,y4的滤波器个数统一

    y1 = Conv2D(filters3, (1, 1), name=conv_name_base + '2d')(y1)
    y2 = Conv2D(filters3, (1, 1), name=conv_name_base + '2e')(y2)
    #concat y1,y2,y3,y4
    y = layers.concatenate([y1, y2, y3, y4], axis=bn_axis)
    y = Conv2D((filters3 * 4), (1, 1), name=conv_name_base + '2')(y)
    y = BatchNormalization(axis=bn_axis, name=bn_name_base + '2')(y)
    t = Lambda(lambda y: tf.split(y, 4, axis=3))(y)
    y = layers.average([t[0], t[1], t[2], t[3]])

    y = layers.multiply([x, y])

    x = layers.add([x, y])
    x = Activation('relu')(x)
    return x



def resnet10(include_top = True,weights=None,input_tensor=None,
           input_shape=None,pooling = 'avg',classes=8):

    #确定合适的输入
    input_shape = _obtain_input_shape(input_shape,default_size=224,min_size=197,
                                      data_format=K.image_data_format,
                                      require_flatten=include_top,weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    #构建网络框架
    #padding='same' 填充,padding='valid'不填充

    x = Conv2D(64,(7,7),strides=2,padding='same',name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis,name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    #构建residual block
    x = conv_block_10(x ,3, [64, 64], stage=2, block='a', strides=(1,1))
    x = conv_block_10(x, 3, [64, 64], stage=2, block='b', strides=(1,1))

    x = conv_block_10(x, 3, [128, 128], stage=3, block='a')
    x = conv_block_10(x, 3, [128, 128], stage=3, block='b', strides=(1, 1))

    x = conv_block_10(x, 3, [256, 256], stage=4, block='a')
    x = conv_block_10(x, 3, [256, 256], stage=4, block='b', strides=(1, 1))

    x = conv_block_10(x, 3, [512, 512], stage=5, block='a')
    x = conv_block_10(x, 3, [512, 512], stage=5, block='b', strides=(1, 1))

    x = AveragePooling2D((7,7), name='avg_pool')(x)

    # x = Flatten()(x)
    # x = Dense(classes,activation='softmax', name='fc8')(x)
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc8')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)



    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input


    # Create model.
    model = Model(inputs, x, name='resnet')

    if weights is not None:
        model.load_weights(weights,by_name=True)
    return model

def resnet10_attention(include_top = True,weights=None,input_tensor=None,
           input_shape=None,pooling = 'avg',classes=8):

    #确定合适的输入
    input_shape = _obtain_input_shape(input_shape,default_size=224,min_size=197,
                                      data_format=K.image_data_format,
                                      require_flatten=include_top,weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    #构建网络框架
    #padding='same' 填充,padding='valid'不填充

    x = Conv2D(64,(7,7),strides=2,padding='same',name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis,name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    #构建residual block
    x = conv_attention_block_10(x ,3, [64, 64], stage=2, block='a', strides=(1,1))
    x = conv_attention_block_10(x, 3, [64, 64], stage=2, block='b', strides=(1,1))

    x = conv_attention_block_10(x, 3, [128, 128], stage=3, block='a')
    x = conv_attention_block_10(x, 3, [128, 128], stage=3, block='b', strides=(1, 1))

    x = conv_attention_block_10(x, 3, [256, 256], stage=4, block='a')
    x = conv_attention_block_10(x, 3, [256, 256], stage=4, block='b', strides=(1, 1))

    x = conv_attention_block_10(x, 3, [512, 512], stage=5, block='a')
    x = conv_attention_block_10(x, 3, [512, 512], stage=5, block='b', strides=(1, 1))

    x = AveragePooling2D((7,7), name='avg_pool')(x)

    # x = Flatten()(x)
    # x = Dense(classes,activation='softmax', name='fc8')(x)
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc8')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)



    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input


    # Create model.
    model = Model(inputs, x, name='resnet')

    if weights is not None:
        model.load_weights(weights)
    return model


@tf_export('keras.applications.ResNet50',
           'keras.applications.resnet50.ResNet50')
def resnet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=5):
  """Instantiates the ResNet50 architecture.

  Optionally loads weights pre-trained
  on ImageNet. Note that when using TensorFlow,
  for best performance you should set
  `image_data_format='channels_last'` in your Keras config
  at ~/.keras/keras.json.

  The model and the weights are compatible with both
  TensorFlow and Theano. The data format
  convention used by the model is the one
  specified in your Keras config file.

  Arguments:
      include_top: whether to include the fully-connected
          layer at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(224, 224, 3)` (with `channels_last` data format)
          or `(3, 224, 224)` (with `channels_first` data format).
          It should have exactly 3 inputs channels,
          and width and height should be no smaller than 197.
          E.g. `(200, 200, 3)` would be one valid value.
      pooling: Optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.

  Returns:
      A Keras model instance.

  Raises:
      ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
  """
  if not (weights in {'imagenet', None} or os.path.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as imagenet with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = _obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=197,
      data_format=K.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = Input(shape=input_shape)
  else:
    if not K.is_keras_tensor(input_tensor):
      img_input = Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1

  x = Conv2D(
      64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
  x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

  x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

  x = AveragePooling2D((7, 7), name='avg_pool')(x)

  if include_top:
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)
  else:
    if pooling == 'avg':
      x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.
  model = Model(inputs, x, name='resnet50')

  # load weights
  if weights == 'imagenet':
    if include_top:
      weights_path = get_file(
          'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
          WEIGHTS_PATH,
          cache_subdir='models',
          md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
    else:
      weights_path = get_file(
          'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
          WEIGHTS_PATH_NO_TOP,
          cache_subdir='models',
          md5_hash='a268eb855778b3df3c7506639542a6af')
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


def resnet50_attention(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=5):
  """Instantiates the ResNet50 architecture.

  Optionally loads weights pre-trained
  on ImageNet. Note that when using TensorFlow,
  for best performance you should set
  `image_data_format='channels_last'` in your Keras config
  at ~/.keras/keras.json.

  The model and the weights are compatible with both
  TensorFlow and Theano. The data format
  convention used by the model is the one
  specified in your Keras config file.

  Arguments:
      include_top: whether to include the fully-connected
          layer at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(224, 224, 3)` (with `channels_last` data format)
          or `(3, 224, 224)` (with `channels_first` data format).
          It should have exactly 3 inputs channels,
          and width and height should be no smaller than 197.
          E.g. `(200, 200, 3)` would be one valid value.
      pooling: Optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.

  Returns:
      A Keras model instance.

  Raises:
      ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
  """
  # Determine proper input shape
  input_shape = _obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=197,
      data_format=K.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = Input(shape=input_shape)
  else:
    if not K.is_keras_tensor(input_tensor):
      img_input = Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor
  if K.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1

  x = Conv2D(
      64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
  x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  x = conv_attention_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

  x = conv_attention_block(x, 3, [128, 128, 512], stage=3, block='a')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

  x = conv_attention_block(x, 3, [256, 256, 1024], stage=4, block='a')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

  x = conv_attention_block(x, 3, [512, 512, 2048], stage=5, block='a')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

  x = AveragePooling2D((7, 7), name='avg_pool')(x)

  if include_top:
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc5')(x)
  else:
    if pooling == 'avg':
      x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.
  model = Model(inputs, x, name='resnet50_attention')

  # load weights
  if weights is not None:
    model.load_weights(weights)

  return model




#model = resnet10_attention(include_top=False)
# plot_model(model,show_shapes=True, to_file='model_attention_10.png')
#model.summary()
#print(model.layers[134].name)
#plot_model(model,show_shapes=True,to_file='model1.png')
