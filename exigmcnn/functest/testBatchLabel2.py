# -*- coding: utf-8 -*-
# @Time    : 2019/4/26 下午2:51
# @Author  : zhuzhengyi

# import pandas as pd
import numpy as np
import tensorflow as tf


def generate_data():
    num = 25
    label = np.asarray(range(0, num))
    images = np.random.random([num, 5])
    print('label size :{}, image size {}'.format(label.shape, images.shape))
    return images,label

def get_batch_data():
    label, images = generate_data()
    input_queue = tf.train.slice_input_producer([images, label], shuffle=False,num_epochs=2)
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=5, num_threads=1, capacity=64,allow_smaller_final_batch=False)
    return image_batch,label_batch


images,label = get_batch_data()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())#就是这一行
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess,coord)
try:
    while not coord.should_stop():
        i,l = sess.run([images,label])
        print(i)
        print(l)
except tf.errors.OutOfRangeError:
    print('Done training')
finally:
    coord.request_stop()
coord.join(threads)
sess.close()
