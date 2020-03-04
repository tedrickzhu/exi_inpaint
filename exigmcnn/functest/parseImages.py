#encoding=utf-8
#author:zhuzhengyi
#time:2019/4/29 下午4:29

from os import listdir,makedirs
from os.path import exists
import cv2

def ResizeImages(dataset_path):
    filelist = listdir(dataset_path)
    if len(filelist)<1:
        return None
    resized_dataset_path=dataset_path+dataset_path.split('/')[-2]+'_parse/'
    print(resized_dataset_path)
    if not exists(resized_dataset_path):
        makedirs(resized_dataset_path)
    for filename in filelist:
        image = cv2.imread(dataset_path+'/'+filename)
        h = image.shape[0]
        w = image.shape[1]
        if w > h:
            image1 = image[:, :h, :]
            image2 = image[:, (w - h):, :]
            cv2.imwrite(resized_dataset_path +filename+ '_1.jpg', image1)
            cv2.imwrite(resized_dataset_path +filename+ '_2.jpg', image2)
        elif h > w:
            image1 = image[:w, :, :]
            image2 = image[(h - w):, :, :]
            cv2.imwrite(resized_dataset_path +filename+ '_1.jpg', image1)
            cv2.imwrite(resized_dataset_path +filename+ '_2.jpg', image2)
            # print('after write')
    return resized_dataset_path

def resizeimage():
    filepath='/data/jpg10/100000.jpg'
    image = cv2.imread(filepath)
    h = image.shape[0]
    w = image.shape[1]
    if w > h:
        image1 = image[:, :h, :]
        image2 = image[:, (w-h):, :]
        cv2.imwrite(filepath + '.1.jpg', image1)
        cv2.imwrite(filepath + '.2.jpg', image2)
    elif h > w:
        image1 = image[:w,:,:]
        image2 = image[(h-w):,:,:]
        cv2.imwrite('/Users/jintaoduan/data/jpg10/xxx/'+'1000.1.jpg',image1)
        cv2.imwrite('/Users/jintaoduan/data/jpg10/xxx/' + '1000.2.jpg', image2)

def CreateSimilartyPairFile(dataset_path):
    filelist = listdir(dataset_path)
    if len(filelist) < 1:
        return None
    code_dict = {}
    for file in filelist:
        grayimage = cv2.imread(dataset_path+file,0)
        code_dict[file] = None


if __name__ == '__main__':
    dataset_path = '/Users/jintaoduan/data/jpg276_100/'
    resized_dataset_path = ResizeImages(dataset_path)
    # resized_dataset_path = '/Users/jintaoduan/PycharmProjects/data/jpg10/jpg10_parse/'


    print('finished!')

