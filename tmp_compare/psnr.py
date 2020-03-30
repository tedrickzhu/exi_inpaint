# -*- coding: utf-8 -*-
# @Time    : 20-3-11 下午9:47
# @Author  : zhuzhengyi
import glob
import os
import tensorflow as tf

# import numpy
# import math
# # import scipy.misc
#
#
# def psnr(img1, img2):
#     mse = numpy.mean( (img1 - img2) ** 2 )
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def read_img(path):
    return tf.image.decode_image(tf.read_file(path))

def tf_psnr(tf_img1, tf_img2):
    return tf.image.psnr(tf_img1, tf_img2, max_val=255)


def caculate():
    t1 = read_img('t1.jpg')
    t2 = read_img('t2.jpg')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y = sess.run(tf_psnr(t1, t2))

def cal_psnr(truthdset,resultset,psnrfile):
    truthfilelist = glob.glob(truthdset + "*.png")
    resfilelist = glob.glob(resultset + "*.png")
    psnrsum = 0.0
    with open(psnrfile,'a+') as txtfile:
        for tru_imgpath in truthfilelist:
            res_imgpath = resultset+os.path.basename(tru_imgpath)
            if res_imgpath in resfilelist:
                t1 = read_img(tru_imgpath)
                t2 = read_img(res_imgpath)
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    y = sess.run(tf_psnr(t1, t2))
                    psnrsum = psnrsum+y
                    txtfile.write(str(os.path.basename(tru_imgpath))+':'+str(y))
                    txtfile.write('\r\n')
        txtfile.write(str(psnrsum/(len(truthfilelist)*1.0)))
        txtfile.write('\r\n')
    txtfile.close()

def ssim_test(truthdset,resultset,ssimfile):

    truthfilelist = glob.glob(truthdset + "*.png")
    resfilelist = glob.glob(resultset + "*.png")
    ssimsum = 0.0
    with open(ssimfile,'a+') as txtfile:
        for tru_imgpath in truthfilelist:
            res_imgpath = resultset+os.path.basename(tru_imgpath)
            if res_imgpath in resfilelist:
                im1 = read_img(tru_imgpath)
                im2 = read_img(res_imgpath)
                # Compute SSIM over tf.uint8 Tensors.
                ssim1 = tf.image.ssim(im1,im2,max_val=255)
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    y = sess.run(ssim1)
                ssimsum = ssimsum+y
                txtfile.write(str(os.path.basename(tru_imgpath))+':'+str(y))
                txtfile.write('\r\n')
        txtfile.write(str(ssimsum/(len(truthfilelist)*1.0)))
        txtfile.write('\r\n')
    txtfile.close()

    # Read images from file.
    # im1 = tf.decode_png('path/to/im1.png')
    # im2 = tf.decode_png('path/to/im2.png')
    # # Compute SSIM over tf.uint8 Tensors.
    # ssim1 = tf.image.ssim(im1,
    #                       im2,
    #                       max_val=255,
    #                       filter_size=11,
    #                       filter_sigma=1.5,
    #                       k1=0.01,
    #                       k2=0.03)

    # # Compute SSIM over tf.float32 Tensors.
    # im1 = tf.image.convert_image_dtype(im1, tf.float32)
    # im2 = tf.image.convert_image_dtype(im2, tf.float32)
    # ssim2 = tf.image.ssim(im1,
    #                       im2,
    #                       max_val=1.0,
    #                       filter_size=11,
    #                       filter_sigma=1.5,
    #                       k1=0.01,
    #                       k2=0.03)

if __name__ == '__main__':
    # truthset = '/home/zzy/work/dnnii_web/dnnii_web/test/tmp_res/context_truth/'
    # resultset = '/home/zzy/work/dnnii_web/dnnii_web/test/tmp_res/context_res/'
    # psnrfile = './context_psnr.txt'
    # cal_psnr(truthset, resultset, psnrfile)
    # ssimfile = './context_ssim.txt'
    # ssim_test(truthset, resultset, ssimfile)

    # truthset = '/home/zzy/TrainData/MITPlace2Dataset/b1000test20/'

    # resultset = '/home/zzy/work/dnnii_web/dnnii_web/test/tmp_res/deepfill_res/'
    # psnrfile = './deepfill_psnr.txt'
    # cal_psnr(truthset, resultset, psnrfile)
    # ssimfile = './deepfill_ssim.txt'
    # ssim_test(truthset, resultset, ssimfile)
    #
    # resultset = '/home/zzy/work/dnnii_web/dnnii_web/test/tmp_res/edge_results/'
    # ssimfile = './edge_ssim.txt'
    # ssim_test(truthset, resultset, ssimfile)
    # # psnrfile = './edge_psnr.txt'
    # # cal_psnr(truthset, resultset, psnrfile)
    # #
    #
    # resultset = '/home/zzy/work/dnnii_web/dnnii_web/test/tmp_res/gmcnn_results/'
    # ssimfile = './gmcnn_ssim.txt'
    # ssim_test(truthset, resultset, ssimfile)
    # # psnrfile = './gmcnn_psnr.txt'
    # # cal_psnr(truthset, resultset, psnrfile)
    # #
    #
    truthset = '/home/zzy/TrainData/MITPlace2Dataset/314-51/'
    # truthset = '/home/zzy/work/exi_inpaint/imgs/places2_512x680/'
    resultset = '/home/zzy/work/exi_inpaint/test_results/exifill314_35/'
    ssimfile = './exifill_ssim0314-51.txt'
    psnrfile = './exifill_psnr0314-51.txt'

    ssim_test(truthset, resultset, ssimfile)
    cal_psnr(truthset, resultset, psnrfile)