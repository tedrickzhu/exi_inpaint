# -*- coding: utf-8 -*-
# @Time    : 20-2-12 上午10:46
# @Author  : zhuzhengyi
import os

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from deepfill.inpaint_model import InpaintCAModel

'''
generative_inpainting 说明：
1，places2的模型，github的介绍说的是使用256x256训练的，但是，测试发现，512x680，256x256的图片均可正常使用
2，celebahq_256,测试得，只可用于256x256的celeba或者celebahq图片，celebahq的效果稍好
3,输入的图片可以是原图，也可以是已经画好待修复区域的图，重要的是mask图片，mask图片中包含有选中区域

'''

def cagc_inp(imagepath,maskpath,checkpointdir,outputpath):
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    # args, unknown = parser.parse_known_args()

    model = InpaintCAModel()
    image = cv2.imread(imagepath)
    mask = cv2.imread(maskpath)
    print('thisisimgshape:',image.shape,mask.shape)
    h, w, _ = image.shape
    mask = cv2.resize(mask, (w,h), fx=0.5, fy=0.5)
    print('thisisimgshape:',image.shape,mask.shape)

    assert image.shape == mask.shape
    #把原始图片划分成grid*grid个格子区域，'//'表示向下取整的除法
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpointdir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        cv2.imwrite(outputpath, result[0][:, :, ::-1])

def cagc_inp_batch():
    baserecutpath = '/home/zzy/TrainData/MITPlace2Dataset/val_recut_512x680'
    imagesnums = 132
    checkpointdir = "./checkpoints/places2_512x680"
    maskpath = "./examples/places2_680x512/wooden_mask.png"
    recutoutput = './examples/recutoutput/'
    recutmaskedoutput = './examples/recutmasked/'

    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    # args, unknown = parser.parse_known_args()
    model = InpaintCAModel()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        for imageindex in range(1, imagesnums):
            imagename = 'Places365_' + str(imageindex).zfill(8) + '.png'
            maskedimagename = 'Places365_' + str(imageindex).zfill(8) + '_masked.png'
            outputimagename = 'Places365_' + str(imageindex).zfill(8) + '_output.png'
            imagepath = os.path.join(baserecutpath, imagename)
            outputpath = os.path.join(recutoutput, outputimagename)
            maskedimagepath = os.path.join(recutmaskedoutput, maskedimagename)

            image = cv2.imread(imagepath)
            mask = cv2.imread(maskpath)
            h, w, _ = image.shape
            mask = cv2.resize(mask, (w, h), fx=0.5, fy=0.5)

            inputimage = image * ((255 - mask) // 255) + mask
            cv2.imwrite(maskedimagepath, inputimage.astype(np.uint8))

            assert image.shape == mask.shape
            # 把原始图片划分成grid*grid个格子区域，'//'表示向下取整的除法
            grid = 8
            image = image[:h // grid * grid, :w // grid * grid, :]
            mask = mask[:h // grid * grid, :w // grid * grid, :]
            print('Shape of image: {}'.format(image.shape))

            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(FLAGS, input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(checkpointdir, from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            print('Model loaded.')
            result = sess.run(output)
            cv2.imwrite(outputpath, result[0][:, :, ::-1])


def ceshi():
    pretrained_models={
        "celebahq":"./checkpoints/celebahq_256",
        "places2":"./checkpoints/places2_512x680"
    }
    checkpointdir=pretrained_models["places2"]

    outputpath="./examples/ceshi512output.png"
    imagepath = "./examples/ceshi512.png"
    maskpath="./examples/places2_680x512/wooden_mask.png"
    # image = cv2.imread(imagepath)
    # mask = cv2.imread(maskpath)
    # print(image.shape,mask.shape)
    cagc_inp(imagepath, maskpath, checkpointdir, outputpath)

    print("this is done")

if __name__ == '__main__':
    # ceshi()
    cagc_inp_batch()