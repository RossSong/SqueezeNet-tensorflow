import numpy as np
import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert(images, targets, tfrecords_filename):

    n_samples = len(targets)
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for j in range(n_samples):

        image = images[j]
        target = targets[j]

        feature = {
            'image_raw': _bytes_feature(image.tostring()),
            'target': _int64_feature(target)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


train_images = np.load('/home/ubuntu/data/train_images.npy')
train_targets = np.load('/home/ubuntu/data/train_targets.npy')

val_images = np.load('/home/ubuntu/data/val_images.npy')
val_targets = np.load('/home/ubuntu/data/val_targets.npy')

convert(train_images, train_targets, 'train.tfrecords')
convert(val_images, val_targets, 'val.tfrecords')
