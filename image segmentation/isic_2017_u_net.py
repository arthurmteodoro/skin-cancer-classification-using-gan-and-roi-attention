# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import keras_unet_collection
from keras_unet_collection import models as unet_models
from keras_unet_collection import losses as unet_losses
import matplotlib.pyplot as plt
import albumentations as A
from tensorflow.keras import backend as K
import glob
import numpy as np
import tensorflow_addons as tfa
import ktrain
import os
import datetime
import segmentation_models as sm

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

def process_images_paths(folder_path):
  image_name = tf.strings.split(folder_path, sep='/')[-1]
  image_name_without_extention = tf.strings.split(image_name, sep='.')[0]

  path_splited = tf.strings.split(folder_path, sep='/')

  path_image = tf.strings.reduce_join([tf.constant(b'/'), path_splited[1], tf.constant(b'/'), 
                                       path_splited[2], tf.constant(b'/'), image_name])
  
  path_lesion = tf.strings.reduce_join([tf.constant(b'/'), path_splited[1], tf.constant(b'/'), 
                                        path_splited[2], tf.constant(b'/'), image_name_without_extention,
                                        tf.constant(b'_segmentation.jpg')])

  return path_image, path_lesion

def read_images(path_image, path_lesion):
  img = tf.io.read_file(path_image)
  lesion = tf.io.read_file(path_lesion)

  img = tf.image.decode_jpeg(img, channels=3)
  lesion = tf.io.decode_jpeg(lesion, channels=1)

  return img, lesion

def read_label(path_image, path_lesion):
  lesion = tf.io.read_file(path_lesion)
  lesion = tf.io.decode_jpeg(lesion, channels=1)

  lesion = tf.cast(lesion, tf.float32) / 255.0

  lesion = lesion[:,:,-1]

  return lesion

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
  if imagenet_norm:
    norm_img = normalize_transform(image=img)['image']
    norm_img = tf.cast(norm_img, tf.float32)
  else:
    norm_img = tf.cast(img, tf.float32) / 255.0
  
  norm_mask = tf.cast(mask, tf.float32) / 255.0

  return norm_img, norm_mask

def norm_fn_np(img, mask, imagenet_norm):
  return tf.numpy_function(normalize_fn, inp=[img, mask, imagenet_norm], Tout=[tf.float32, tf.float32])

def stretch_pre(nimg):
    """
    from 'Applicability Of White-Balancing Algorithms to Restoring Faded Colour Slides: An Empirical Evaluation'
    """
    nimg = nimg.transpose(2, 0, 1)
    nimg[0] = np.maximum(nimg[0]-nimg[0].min(),0)
    nimg[1] = np.maximum(nimg[1]-nimg[1].min(),0)
    nimg[2] = np.maximum(nimg[2]-nimg[2].min(),0)
    return nimg.transpose(1, 2, 0)

def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return  nimg.transpose(1, 2, 0).astype(np.uint8)

def max_white(nimg):
    if nimg.dtype==np.uint8:
        brightest=float(2**8)
    elif nimg.dtype==np.uint16:
        brightest=float(2**16)
    elif nimg.dtype==np.uint32:
        brightest=float(2**32)
    else:
        brightest==float(2**8)
    nimg = nimg.transpose(2, 0, 1)
    nimg = nimg.astype(np.int32)
    nimg[0] = np.minimum(nimg[0] * (brightest/float(nimg[0].max())),255)
    nimg[1] = np.minimum(nimg[1] * (brightest/float(nimg[1].max())),255)
    nimg[2] = np.minimum(nimg[2] * (brightest/float(nimg[2].max())),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def stretch(nimg):
    return max_white(stretch_pre(nimg))

def retinex(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = nimg[1].max()
    nimg[0] = np.minimum(nimg[0]*(mu_g/float(nimg[0].max())),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/float(nimg[2].max())),255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinex_adjust(nimg):
    """
    from 'Combining Gray World and Retinex Theory for Automatic White Balance in Digital Photography'
    """
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    sum_r = np.sum(nimg[0])
    sum_r2 = np.sum(nimg[0]**2)
    max_r = nimg[0].max()
    max_r2 = max_r**2
    sum_g = np.sum(nimg[1])
    max_g = nimg[1].max()
    coefficient = np.linalg.solve(np.array([[sum_r2,sum_r],[max_r2,max_r]]),
                                  np.array([sum_g,max_g]))
    nimg[0] = np.minimum((nimg[0]**2)*coefficient[0] + nimg[0]*coefficient[1],255)
    sum_b = np.sum(nimg[1])
    sum_b2 = np.sum(nimg[1]**2)
    max_b = nimg[1].max()
    max_b2 = max_r**2
    coefficient = np.linalg.solve(np.array([[sum_b2,sum_b],[max_b2,max_b]]),
                                             np.array([sum_g,max_g]))
    nimg[1] = np.minimum((nimg[1]**2)*coefficient[0] + nimg[1]*coefficient[1],255)
    return nimg.transpose(1, 2, 0).astype(np.uint8)

def retinex_with_adjust(nimg):
    return retinex_adjust(retinex(nimg))

def color_process_fn(img, method):
  if method is 'stretch':
    return tf.numpy_function(stretch, inp=[img], Tout=tf.uint8)
  elif method is 'grey_world':
    return tf.numpy_function(grey_world, inp=[img], Tout=tf.uint8)
  elif method is 'retinex':
    return tf.numpy_function(retinex, inp=[img], Tout=tf.uint8)
  elif method is 'max_white':
    return tf.numpy_function(max_white, inp=[img], Tout=tf.uint8)
  elif method is 'retinex_adjust':
    return tf.numpy_function(retinex_adjust, inp=[img], Tout=tf.uint8)
  else:
    return img

def process_images(dermatoscopic, lesion, aug=True, color=None, imagenet_norm=True):
  dermatoscopic_shape = tf.shape(dermatoscopic)
  lesion_shape = tf.shape(lesion)

  if color is not None:
    dermatoscopic = color_process_fn(dermatoscopic, color)

  if aug:
    dermatoscopic, lesion = aug_fn_np(dermatoscopic, lesion)

  dermatoscopic = tf.reshape(dermatoscopic, dermatoscopic_shape)
  lesion = tf.reshape(lesion, lesion_shape)

  lesion = tf.cast(lesion, tf.float32) / 255.0

  return dermatoscopic, lesion

def get_dataset(files_path, augmented=True, shuffle=True, color=None):
  ds = tf.data.Dataset.list_files(files_path, shuffle=False)

  ds = ds.map(process_images_paths)

  ds = ds.map(read_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  ds = ds.cache()

  ds = ds.map(lambda img, mask: process_images(img, mask, augmented, color), 
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if shuffle:
    ds = ds.shuffle(BUFFER_SIZE)
    
  ds = ds.batch(BATCH_SIZE).repeat()
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return ds

images_names = os.listdir('/content/ISIC_2017_NO_HAIR/')
images_names = [x for x in images_names if 'segmentation' not in x]
train_files = [os.path.join('/content/ISIC_2017_NO_HAIR', x) for x in images_names]

images_names = os.listdir('/content/ISIC_2017_VALIDATION/')
images_names = [x for x in images_names if 'segmentation' not in x]
validation_files = [os.path.join('/content/ISIC_2017_VALIDATION', x) for x in images_names]

images_names = os.listdir('/content/ISIC_2017_TEST/')
images_names = [x for x in images_names if 'segmentation' not in x]
test_files = [os.path.join('/content/ISIC_2017_TEST', x) for x in images_names]

BATCH_SIZE = 8
BUFFER_SIZE = 1000

train_ds = get_dataset(train_files)
validation_ds = get_dataset(validation_files, augmented=False, shuffle=False)
test_ds = get_dataset(test_files, augmented=False, shuffle=False)

for image, mask in train_ds.take(1):
  sample_image, sample_mask = image[0], mask[0]
display([sample_image, sample_mask])

for image, mask in validation_ds.take(1):
  sample_image, sample_mask = image[0], mask[0]
display([sample_image, sample_mask])

for image, mask in test_ds.take(1):
  sample_image, sample_mask = image[0], mask[0]
display([sample_image, sample_mask])

STEPS_PER_EPOCH = len(train_files) // BATCH_SIZE
STEPS_PER_EPOCH_VALIDATION = len(validation_files) // BATCH_SIZE
STEPS_PER_EPOCH_TEST = len(test_files) // BATCH_SIZE

def get_labels(files_path, augmented=True, shuffle=True):
  ds = tf.data.Dataset.list_files(files_path, shuffle=False)
  ds = ds.map(process_images_paths)
  ds = ds.map(read_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return ds

training_labels = list(get_labels(train_files).as_numpy_iterator())
validation_labels = list(get_labels(validation_files).as_numpy_iterator())

trn = ktrain.TFDataset(train_ds, n=len(training_labels), y=training_labels)
val = ktrain.TFDataset(validation_ds, n=len(validation_labels), y=validation_labels)

model = unet_models.unet_2d((256, 256, 3), [64, 128, 256, 512, 1024], n_labels=1,
                       stack_num_down=3, stack_num_up=2,
                       activation='ReLU', output_activation='Sigmoid',
                       backbone='EfficientNetB7', freeze_backbone=False,
                       batch_norm=True, pool=True, unpool=True)
x = tf.keras.layers.Reshape((256, 256))(model.output)
model = tf.keras.Model(inputs=model.input, outputs=x)
model.summary()


def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis = -1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))

loss = jaccard_distance#unet_losses.focal_tversky
opt = tfa.optimizers.AdamW(1e-5, 1e-3)

iou = sm.metrics.IOUScore(name='iou', threshold=0.5)
dice_coe = sm.metrics.FScore(name='dice_coe', threshold=0.5)
precision = sm.metrics.Precision(name='precision', threshold=0.5)
recall = sm.metrics.Recall(name='recall', threshold=0.5)

model.compile(optimizer=opt, loss=loss,
              metrics=[iou, dice_coe, precision, recall, accuracy])

learner = ktrain.get_learner(model, train_data=trn, val_data=val)

class SaveBestModelTestDataset(tf.keras.callbacks.Callback):
  def __init__(self, test_ds, num_steps, filename, metric):
    super(SaveBestModelTestDataset, self).__init__()
    self.test = test_ds
    self.num_steps = num_steps
    self.filename = filename
    self.best_value = 0
    self.metric = metric


  def on_epoch_end(self, epoch, logs=None):
      results = self.model.evaluate(self.test, steps=self.num_steps, return_dict=True, verbose=0)
      cur_metric = results[self.metric]
      if cur_metric > self.best_value:
        print('\nEpoch %d: %s improved from %.6f to %.6f, saving model to %s' % (epoch+1, self.metric, self.best_value, cur_metric, self.filename))
        self.model.save(self.filename)
        self.best_value = cur_metric
      else:
        print('\nEpoch %d: %s did not improve from %.6f' % (epoch+1, self.metric, self.best_value))

ch1 = SaveBestModelTestDataset(test_ds, STEPS_PER_EPOCH_TEST, 'best_iou.h5', 'iou')
ch2 = SaveBestModelTestDataset(test_ds, STEPS_PER_EPOCH_TEST, 'best_dice.h5', 'dice_coe')

ch3 = CheckLR()
ch4 = CheckTestMetrics(test_ds, STEPS_PER_EPOCH_TEST)

model_history = learner.fit_onecycle(1e-3, 150, callbacks=[ch1, ch2, ch3, ch4])

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

learner.plot('lr')

from keras_unet_collection.activations import GELU

model = tf.keras.models.load_model('best_iou.h5', custom_objects={'focal_tversky': unet_losses.focal_tversky,
                                                                  'iou': iou,
                                                                  'dice_coe': dice_coe,
                                                                  'precision': precision,
                                                                  'recall': recall,
                                                                  'accuracy': accuracy,
                                                                  'GELU': GELU}, compile=False)

loss = unet_losses.focal_tversky#jaccard_distance
opt = tfa.optimizers.AdamW(1e-5, 1e-3)

iou = sm.metrics.IOUScore(name='iou')
dice_coe = sm.metrics.FScore(name='dice_coe')
precision = sm.metrics.Precision(name='precision')
recall = sm.metrics.Recall(name='recall')

model.compile(optimizer=opt, loss=loss,
              metrics=[iou, dice_coe, precision, recall, accuracy])

for image, mask in test_ds.take(1):
  sample_image, sample_mask = image[0], mask[0]

y_pred = model.predict(sample_image[tf.newaxis, ...])
y_pred

plt.figure(figsize=(16, 16))
plt.subplot(1,3,1)
plt.imshow(sample_image)
plt.title('Original Image')
plt.subplot(1,3,2)
#plt.imshow(sample_mask, plt.cm.binary_r)
plt.imshow(sample_mask[:,:,-1], plt.cm.binary_r)
plt.title('Ground Truth')
plt.subplot(1,3,3)
plt.imshow(np.around(y_pred.reshape(256, 256)), plt.cm.binary_r)
plt.title('Predicted Output')
plt.show()

results = model.evaluate(test_ds, steps=STEPS_PER_EPOCH_TEST, return_dict=True)

print('Best IoU')
print('Loss: %s' % format(results['loss'], '.6f').replace('.', ','))
print('IoU: %s' % format(results['iou'], '.6f').replace('.', ','))
print('Dice: %s' % format(results['dice_coe'], '.6f').replace('.', ','))
print('Precision: %s' % format(results['precision'], '.6f').replace('.', ','))
print('Recall: %s' % format(results['recall'], '.6f').replace('.', ','))
print('Accuracy: %s' % format(results['accuracy'], '.6f').replace('.', ','))

from keras_unet_collection.activations import GELU

model = tf.keras.models.load_model('best_dice.h5', custom_objects={'focal_tversky': unet_losses.focal_tversky,
                                                                  'iou': iou,
                                                                  'dice_coe': dice_coe,
                                                                  'precision': precision,
                                                                  'recall': recall,
                                                                  'accuracy': accuracy,
                                                                  'GELU': GELU}, compile=False)

loss = unet_losses.focal_tversky#jaccard_distance
opt = tfa.optimizers.AdamW(1e-5, 1e-3)

iou = sm.metrics.IOUScore(name='iou')
dice_coe = sm.metrics.FScore(name='dice_coe')
precision = sm.metrics.Precision(name='precision')
recall = sm.metrics.Recall(name='recall')

model.compile(optimizer=opt, loss=loss,
              metrics=[iou, dice_coe, precision, recall, accuracy])

y_pred = model.predict(sample_image[tf.newaxis, ...])

plt.figure(figsize=(16, 16))
plt.subplot(1,3,1)
plt.imshow(sample_image)
plt.title('Original Image')
plt.subplot(1,3,2)
#plt.imshow(sample_mask, plt.cm.binary_r)
plt.imshow(sample_mask[:,:,-1], plt.cm.binary_r)
plt.title('Ground Truth')
plt.subplot(1,3,3)
plt.imshow(np.around(y_pred.reshape(256, 256)), plt.cm.binary_r)
plt.title('Predicted Output')
plt.show()

results = model.evaluate(test_ds, steps=STEPS_PER_EPOCH_TEST, return_dict=True)

print('Best Dice')
print('Loss: %s' % format(results['loss'], '.6f').replace('.', ','))
print('IoU: %s' % format(results['iou'], '.6f').replace('.', ','))
print('Dice: %s' % format(results['dice_coe'], '.6f').replace('.', ','))
print('Precision: %s' % format(results['precision'], '.6f').replace('.', ','))
print('Recall: %s' % format(results['recall'], '.6f').replace('.', ','))
print('Accuracy: %s' % format(results['accuracy'], '.6f').replace('.', ','))