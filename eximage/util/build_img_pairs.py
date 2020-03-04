#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 19-11-29 下午2:36
# @Author  : zhuzhengyi
# @File    : searchengine.py
# @Software: PyCharm

from searcher import structuredescriptor, searcher, colordescriptor
import cv2
import glob

def buildimgpairs(dataset,colorIndexPath,structureIndexPath,result):

	with open(result, "w") as output:
		# # 匹配结果存储
		for imagePath in glob.glob(dataset + "*.png"):
			imageName = imagePath.split("/")[-1].split(".")[0]
			imageSearcher = searcher.Searcher(colorIndexPath, structureIndexPath)
			queryFeatures = None
			queryStructures = None

			idealBins = (8, 12, 3)
			idealDimension = (16, 16)

			# 传入色彩空间的bins
			colorDescriptor = colordescriptor.ColorDescriptor(idealBins)
			# 传入构图空间的bins
			structureDescriptor = structuredescriptor.StructureDescriptor(idealDimension)
			queryImage = cv2.imread(imagePath)
			queryFeatures = colorDescriptor.describe(queryImage)
			queryStructures = structureDescriptor.describe(queryImage)
			###########################################
			#检索
			searchResults = imageSearcher.search(queryFeatures, queryStructures)
			respairs = []
			for resimgname, score in searchResults:
				if resimgname!=imageName:
					respairs.append(resimgname)
			# 将匹配结果写入到csv文件中去
			output.write("%s,%s\n" % (imageName, ",".join(respairs)))
	# close index file
	output.close()

if __name__ == '__main__':
	# dataset = '/home/zzy/TrainData/MITPlace2Dataset/val_recut_512x680/'
	dataset = '/home/zzy/work/experiment_image_inpaint/searcher/examples/output/'
	# dataset = "/home/zzy/TrainData/HolidaySet/"
	colorIndexPath = "/home/zzy/work/experiment_image_inpaint/searcher/colorfeatures.csv"
	structureIndexPath = "/home/zzy/work/experiment_image_inpaint/searcher/strucfeatures.csv"
	# result = "/home/zzy/work/experiment_image_inpaint/searcher/respairs000.csv"
	result = "/home/zzy/work/experiment_image_inpaint/searcher/respairs111.csv"
	buildimgpairs(dataset, colorIndexPath, structureIndexPath, result)
	print('build image pairs done.')