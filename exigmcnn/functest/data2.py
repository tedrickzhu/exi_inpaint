# -*- coding: utf-8 -*-
# @Time    : 20-3-5 上午12:07
# @Author  : zhuzhengyi

import tensorflow as tf

class DataLoader:
    def __init__(self, filename, im_size, batch_size):
        self.filelist,self.simfilelist = self.get_filename_pair(filename)

        if not self.filelist:
            exit('\nError: file list is empty\n')
        
        self.im_size = im_size
        self.batch_size = batch_size
        self.data_queue = None

    def get_filename_pair(self,filename):
        linelist = open(filename, 'rt').read().splitlines()
        filelist = []
        simfilelist = []
        for line in linelist:
            names = line.split(',')
            filelist.append(names[0])
            simfilelist.append(names[1])
        return filelist,simfilelist

    def next(self):
        with tf.variable_scope('feed'):
            filelist_tensor = tf.convert_to_tensor(self.filelist, dtype=tf.string)
            simfilelist_tensor = tf.convert_to_tensor(self.simfilelist, dtype=tf.string)

            # self.data_queue ,= tf.train.slice_input_producer([filelist_tensor])
            # self.simdata_queue = tf.train.slice_input_producer([simfilelist_tensor])
            self.data_queue ,= tf.train.slice_input_producer([filelist_tensor],shuffle=False)
            self.simdata_queue = tf.train.slice_input_producer([simfilelist_tensor],shuffle=False)
            # print(type(self.data_queue),type(self.simdata_queue))
            # #this code is only a variable_scope,there are no value,so can't print
            # for i in range(len(self.data_queue)):
            #     print(self.data_queue[i],self.simdata_queue[i])

            im_gt = tf.image.decode_image(tf.read_file(self.data_queue[0]), channels=3)
            sim_im_gt = tf.image.decode_image(tf.read_file(self.simdata_queue[0]), channels=3)
            # im_gt = tf.cast(im_gt, tf.float32) / 127.5 - 1
            im_gt = tf.cast(im_gt, tf.float32)
            sim_im_gt = tf.cast(sim_im_gt, tf.float32)

            im_gt = tf.image.resize_image_with_crop_or_pad(im_gt, self.im_size[0], self.im_size[1])
            sim_im_gt = tf.image.resize_image_with_crop_or_pad(sim_im_gt, self.im_size[0], self.im_size[1])

            im_gt.set_shape([self.im_size[0], self.im_size[1], 3])
            sim_im_gt.set_shape([self.im_size[0], self.im_size[1], 3])

            batch_gt,sim_batch = tf.train.batch([im_gt,sim_im_gt], batch_size=self.batch_size, num_threads=4)
        return batch_gt
