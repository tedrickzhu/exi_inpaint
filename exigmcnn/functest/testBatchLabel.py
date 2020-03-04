# -*- coding: utf-8 -*-
# @Time    : 20-3-5 上午12:07
# @Author  : zhuzhengyi

import tensorflow as tf
import numpy as np


def generate_data():
    num = 25
    label = np.asarray(range(0, num))
    # images = np.random.random([num, 5, 5, 3])
    images = np.asarray(range(0, num))
    print('label size :{}, image size {}'.format(label.shape, images.shape))
    return label, images


def get_batch_data():
    label, images = generate_data()
    images = tf.cast(images, tf.float32)
    label = tf.cast(label, tf.int32)
    #下面两种方式均可以保证images和label的对应关系不会混乱，区别在于，不shuffle时，取出的顺序和原数组中的顺序是一致的
    input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
    # input_queue = tf.train.slice_input_producer([images, label])
    #该方法会导致images和label的对应关系混乱
    # input_data_queue = tf.train.slice_input_producer([images])
    # input_label_queue = tf.train.slice_input_producer([label])
    #固定两个取出顺序均按照原数组的顺序，以此来保证images和label的对应关系
    # input_data_queue = tf.train.slice_input_producer([images], shuffle=False)
    # input_label_queue = tf.train.slice_input_producer([label], shuffle=False)
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=10, num_threads=1, capacity=64)
    # image_batch, label_batch = tf.train.batch([input_data_queue,input_label_queue], batch_size=10, num_threads=1, capacity=64)
    return image_batch, label_batch


image_batch, label_batch = get_batch_data()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    i = 0
    try:
        while not coord.should_stop():
            image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
            i += 1
            for j in range(10):
                print(image_batch_v[j], label_batch_v[j])
    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)