#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 19-11-29 下午2:37
# @Author  : zhuzhengyi
# @File    : index.py
# @Software: PyCharm

from searcher.colordescriptor import ColorDescriptor
from searcher.structuredescriptor import StructureDescriptor
import glob
import cv2

def buildfeaturesdb(dataset,colorfeas,strucfeas):
	idealBins = (8, 12, 3)
	colorDesriptor = ColorDescriptor(idealBins)

	with open(colorfeas, "w") as output:
		# 色彩空间的特征存储
		for imagePath in glob.glob(dataset + "*.png"):
			imageName = imagePath.split("/")[-1].split(".")[0]
			image = cv2.imread(imagePath)
			features = colorDesriptor.describe(image)
			# 将色彩空间的特征写入到csv文件中去
			features = [str(feature).replace("\n", "") for feature in features]
			output.write("%s,%s\n" % (imageName, ",".join(features)))
		# close index file
	output.close()

	idealDimension = (16, 16)
	structureDescriptor = StructureDescriptor(idealDimension)

	with open(strucfeas, "w") as output:
		# 构图空间的色彩特征存储
		for imagePath in glob.glob(dataset + "*.png"):
			imageName = imagePath.split("/")[-1].split(".")[0]
			image = cv2.imread(imagePath)
			structures = structureDescriptor.describe(image)
			# 将构图空间的色彩特征写入到文件中去  write structures to file
			structures = [str(structure).replace("\n", "") for structure in structures]
			output.write("%s,%s\n" % (imageName, ",".join(structures)))
		# close index file
	output.close()

if __name__ == '__main__':
	dataset = '/home/zzy/TrainData/MITPlace2Dataset/val_recut_512x680/'
	# dataset = "/home/zzy/TrainData/HolidaySet/"
	colorfeas = "/home/zzy/work/experiment_image_inpaint/searcher/colorfeatures.csv"
	strucfeas = "/home/zzy/work/experiment_image_inpaint/searcher/strucfeatures.csv"
	buildfeaturesdb(dataset, colorfeas, strucfeas)
	print('build features file done.')