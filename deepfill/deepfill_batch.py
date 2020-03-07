import glob
import time
import os

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from deepfill.inpaint_model import InpaintCAModel


# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--flist', default='', type=str,
#     help='The filenames of image to be processed: input, mask, output.')
# parser.add_argument(
#     '--image_height', default=-1, type=int,
#     help='The height of images should be defined, otherwise batch mode is not'
#     ' supported.')
# parser.add_argument(
#     '--image_width', default=-1, type=int,
#     help='The width of images should be defined, otherwise batch mode is not'
#     ' supported.')
# parser.add_argument(
#     '--checkpoint_dir', default='', type=str,
#     help='The directory of tensorflow checkpoint.')

'''
img_mask_txt ,one txt file path,which each line is consist of imagefullpath and maskfullpath
'''
def deepfillbatch(image_height,image_width,checkpoint_dir,img_mask_txt,outputdir):
    FLAGS = ng.Config('./deepfill/inpaint.yml')
    ng.get_gpus(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] =''

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1,  image_height,  image_width*2, 3))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
             checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')
    t = time.time()
    with open(img_mask_txt, 'r') as f:
        while True:
            line = f.readline()  # 整行读取数据
            if not line:
                break

            imagepath, maskpath = line.replace('\n', '').replace('\r', '').split(',')
            print(imagepath)
            print(maskpath)
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
            outputpath = os.path.join(outputdir,os.path.basename(imagepath))

            image = cv2.imread(imagepath)
            mask = cv2.imread(maskpath)
            print(image.shape)
            print(mask.shape)
            image = cv2.resize(image, ( image_width,  image_height))
            mask = cv2.resize(mask, ( image_width,  image_height))
            # cv2.imwrite(maskedimg, image*(1-mask/255.) + mask)
            # # continue
            # image = np.zeros((128, 256, 3))
            # mask = np.zeros((128, 256, 3))

            assert image.shape == mask.shape

            h, w, _ = image.shape
            grid = 4
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :]
            print('Shape of image: {}'.format(image.shape))

            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)

            # load pretrained model
            result = sess.run(output, feed_dict={input_image_ph: input_image})
            print('Processed: {}'.format(outputpath))
            cv2.imwrite(outputpath, result[0][:, :, ::-1])

    print('Time total: {}'.format(time.time() - t))


def create_data_mask_file(datasetdir,maskpath,outputpath):

    with open(outputpath,'a+') as txtfile:
        # for index in range(1,imagesnums):
        for imagepath in glob.glob(datasetdir + "*.png"):
            # imagename = 'Places365_' + str(index).zfill(8) + '.png'
            # imagepath = os.path.join(baserecutpath, imagename)
            line = str(imagepath)+','+str(maskpath)
            txtfile.write(line)
            txtfile.write('\r\n')
    txtfile.close()
    print('finished')



if __name__ == '__main__':
    create_data_mask_file()
    # checkpointdir = "/home/zzy/work/experiment_image_inpaint/searcher/checkpoints/places2_512x680"
    # img_mask_txt_path = './deepfill/img_mask_path.txt'
    # outputdir = '/home/zzy/work/experiment_image_inpaint/searcher/examples/output'
    # deepfillbatch(512,680,checkpointdir,img_mask_txt_path,outputdir)
    # mask = cv2.imread('/home/zzy/work/experiment_image_inpaint/searcher/examples/center_mask_512x680.png')
    # print(mask.shape)