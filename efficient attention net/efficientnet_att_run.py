# -*- coding: utf-8 -*-
from google.colab import drive
drive.mount('/content/drive')

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import albumentations as A
from tensorflow.keras import backend as K
import glob
import numpy as np
import tensorflow_addons as tfa
import ktrain
import os

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    if i == 1:
      img_shape = list(display_list[i].shape)
      if len(img_shape) == 3:
        plt.imshow(display_list[i][:,:,-1], plt.cm.binary_r)
      else:
        plt.imshow(display_list[i], plt.cm.binary_r)
    else:
      plt.imshow(display_list[i])
    plt.axis('off')
  plt.show()

original_images_benign = os.listdir('/content/skin_cancer_v2_no_hair_with_mask/benign')
original_images_benign = [x for x in original_images_benign if 'mask' not in x]
original_images_benign = [os.path.join('/content/skin_cancer_v2_no_hair_with_mask/benign', x) for x in original_images_benign]

original_images_malignant = os.listdir('/content/skin_cancer_v2_no_hair_with_mask/malignant')
original_images_malignant = [x for x in original_images_malignant if 'mask' not in x]
original_images_malignant = [os.path.join('/content/skin_cancer_v2_no_hair_with_mask/malignant', x) for x in original_images_malignant]

total_images_benign = len(original_images_benign)
total_images_malignant = len(original_images_malignant)

#test_index_benign = int(total_images_benign * 0.2)
test_index_malignant = int(total_images_malignant * 0.2)
test_index_benign = test_index_malignant

malignant_train_images = original_images_malignant[test_index_malignant:]
malignant_test_images = original_images_malignant[0:test_index_malignant]

benign_train_images = original_images_benign[test_index_benign:]
benign_test_images = original_images_benign[0:test_index_benign]

original_train_images = benign_train_images + malignant_train_images
original_test_images = benign_test_images + malignant_test_images

np.random.shuffle(original_train_images)
np.random.shuffle(original_test_images)

def process_images_paths(folder_path):
  image_name = tf.strings.split(folder_path, sep='/')[-1]
  image_name_without_extention = tf.strings.split(image_name, sep='.')[0]

  path_splited = tf.strings.split(folder_path, sep='/')

  path_image = tf.strings.reduce_join([tf.constant(b'/'), path_splited[1], tf.constant(b'/'), 
                                       path_splited[2], tf.constant(b'/'), path_splited[3],
                                       tf.constant(b'/'), image_name])
  
  path_lesion = tf.strings.reduce_join([tf.constant(b'/'), path_splited[1], tf.constant(b'/'), 
                                        path_splited[2], tf.constant(b'/'), path_splited[3],
                                        tf.constant(b'/'), image_name_without_extention,
                                        tf.constant(b'_mask.jpg')])
  
  label = tf.where(path_splited[3] == b'benign', 0, 1)

  return (path_image, path_lesion), label

def read_images(image_path, label):
  img = tf.io.read_file(image_path[0])
  img = tf.image.decode_jpeg(img, channels=3)

  mask = tf.io.read_file(image_path[1])
  mask = tf.io.decode_jpeg(mask, channels=1)

  return (img, mask), label

def read_label(image_path):
  path_class = tf.strings.split(image_path, '/')[-2]
  label = tf.where(path_class == b'benign', 0, 1)

  return label

transforms = A.Compose([
                        A.RandomRotate90(),
                        A.HorizontalFlip(),
                        A.VerticalFlip(),
                        A.Transpose(),
                        A.ShiftScaleRotate(),
                        A.GridDistortion()
])

normalize_transform = A.Normalize()

def aug_fn(img, mask):
  aug_data = transforms(image=img, mask=mask)

  img_aug = aug_data['image']
  mask_aug = aug_data['mask']

  img_aug = tf.cast(img_aug, tf.uint8)
  mask_aug = tf.cast(mask_aug, tf.uint8)

  return img_aug, mask_aug

def aug_fn_np(img, mask):
  return tf.numpy_function(aug_fn, inp=[img, mask], Tout=[tf.uint8, tf.uint8])

def normalize_fn(img, mask, imagenet_norm=True):
  #if imagenet_norm:
  #  norm_img = normalize_transform(image=img)['image']
  #  norm_img = tf.cast(norm_img, tf.float32)
  #else:
  #  norm_img = tf.cast(img, tf.float32) / 255.0
  
  norm_img = tf.cast(img, tf.float32) / 255.0
  norm_mask = tf.cast(mask, tf.float32) / 255.0

  return norm_img, norm_mask

def norm_fn_np(img, mask, imagenet_norm):
  return tf.numpy_function(normalize_fn, inp=[img, mask, imagenet_norm], Tout=[tf.float32, tf.float32])

def process_images(imgs, label, aug=True, imagenet_norm=True):
  img_shape = tf.shape(imgs[0])
  mask_shape = tf.shape(imgs[1])

  img = imgs[0]
  mask = imgs[1]

  if aug:
    img, mask = aug_fn_np(imgs[0], imgs[1])

  img = tf.cast(img, tf.float32) / 255.0
  mask = tf.cast(mask, tf.float32) / 255.0
  #img, mask = norm_fn_np(imgs[0], imgs[1], imagenet_norm)

  img = tf.reshape(img, img_shape)
  mask = tf.reshape(mask, mask_shape)

  img = tf.image.resize(img, (224, 224))
  mask = tf.image.resize(mask, (224, 224))

  return (img, mask), label

def get_dataset(files_path, buffer_size, batch_size, augmented=True, 
                shuffle=True, color=None, cache=None, repeat=True):
  ds = tf.data.Dataset.from_tensor_slices(files_path)
  ds = ds.map(process_images_paths, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(read_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if cache is not None:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  if repeat:
    ds = ds.repeat()

  ds = ds.map(lambda img, label: process_images(img, label, augmented, color), 
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  if shuffle:
    ds = ds.shuffle(buffer_size)
    
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return ds

BATCH_SIZE = 32
BUFFER_SIZE = 512

train_ds = get_dataset(original_train_images, buffer_size=BUFFER_SIZE, 
                       batch_size=BATCH_SIZE, cache=True)
test_ds = get_dataset(original_test_images, augmented=False, shuffle=False, 
                      buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, 
                      repeat=False, cache=True)

for images, label in train_ds.take(1):
  sample_image, sample_mask = images[0][0], images[1][0]
display([sample_image, sample_mask])

for images, label in test_ds.take(1):
  sample_image, sample_mask = images[0][0], images[1][0]
display([sample_image, sample_mask])

def get_labels(files_path):
  ds = tf.data.Dataset.from_tensor_slices(files_path)
  ds = ds.map(read_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return ds

training_labels = list(get_labels(original_train_images).as_numpy_iterator())
validation_labels = list(get_labels(original_test_images).as_numpy_iterator())

trn = ktrain.TFDataset(train_ds, n=len(training_labels), y=training_labels)
val = ktrain.TFDataset(test_ds, n=len(validation_labels), y=validation_labels)

from eff_att_v3 import get_att_eff_b0

model = get_att_eff_b0()
model.summary()

from tensorflow.keras import backend as K
import sklearn

def recall_sk(y_true, y_pred):
  y_pred = np.round(y_pred)
  return sklearn.metrics.recall_score(y_true, y_pred, zero_division=0)

def recall(y_true, y_pred):
  return tf.numpy_function(recall_sk, inp=[y_true, y_pred], Tout=tf.float64)

def precision_sk(y_true, y_pred):
  y_pred = np.round(y_pred)
  return sklearn.metrics.precision_score(y_true, y_pred, zero_division=0)

def precision(y_true, y_pred):
  return tf.numpy_function(precision_sk, inp=[y_true, y_pred], Tout=tf.float64)

def bacc_sk(y_true, y_pred):
  y_pred = np.round(y_pred)
  return sklearn.metrics.balanced_accuracy_score(y_true, y_pred)

def bacc(y_true, y_pred):
  return tf.numpy_function(bacc_sk, inp=[y_true, y_pred], Tout=tf.float64)

def roc_auc_sk(y_true, y_pred):
  y_pred = np.round(y_pred)
  min_value_y_true = np.min(y_true)
  max_value_y_true = np.max(y_true)
  if min_value_y_true == max_value_y_true:
    return 0.0
  else:
    return sklearn.metrics.roc_auc_score(y_true, y_pred)

def roc_auc(y_true, y_pred):
  return tf.numpy_function(roc_auc_sk, inp=[y_true, y_pred], Tout=tf.float64)

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
opt = tfa.optimizers.AdamW(weight_decay=1e-5, learning_rate=1e-4)

model.compile(optimizer=opt, loss=loss, 
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                       recall, precision, bacc, roc_auc])

learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=BATCH_SIZE, eval_batch_size=BATCH_SIZE)

from sklearn.utils import class_weight

class_weight_data = class_weight.compute_class_weight('balanced', np.unique(training_labels), training_labels)
class_weight_data = dict(enumerate(class_weight_data))
print(class_weight_data)

import math

class CMCallback(tf.keras.callbacks.Callback):
  def __init__(self, y, y_true):
    super(CMCallback, self).__init__()
    self.test_ds = y
    self.y_true = y_true
    #self.steps = math.ceil(len(y_true) / BATCH_SIZE)
  def on_epoch_end(self, epoch, logs=None):
    y_pred = self.model.predict(self.test_ds)#, steps=self.steps)
    y_pred = np.around(y_pred)
    cm = sklearn.metrics.confusion_matrix(self.y_true, y_pred)
    print('\nConfusion matrix in epoch %d' % epoch)
    print(cm)
    print('')

ch1 = tf.keras.callbacks.ModelCheckpoint('best_auc.h5', monitor='val_roc_auc', verbose=1, save_best_only=True, mode='max')
ch2 = CMCallback(test_ds, validation_labels)
ch3 = tf.keras.callbacks.ModelCheckpoint('best_auc_weights.h5', monitor='val_roc_auc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

model_history = learner.fit_onecycle(1e-2, 30, class_weight=class_weight_data, callbacks=[ch1, ch2, ch3])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']

epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'm', label='Validation accuracy')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

model.load_weights('/content/best_auc_weights.h5')

best_model = model
y_pred = best_model.predict(test_ds)
y_pred = np.where(y_pred >= 0.5, 1, 0)

y_true = validation_labels

print('Accuracy %f' % sklearn.metrics.accuracy_score(y_true, y_pred))
print('Recall %f' % sklearn.metrics.recall_score(y_true, y_pred))
print('Precision %f' % sklearn.metrics.precision_score(y_true, y_pred))
print('AUC %f' % sklearn.metrics.roc_auc_score(y_true, y_pred))
print('CM')
print(sklearn.metrics.confusion_matrix(y_true, y_pred))