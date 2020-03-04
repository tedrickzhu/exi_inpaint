#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 19-11-29 下午2:36
# @Author  : zhuzhengyi
# @File    : searchengine.py
# @Software: PyCharm
from os.path import join
from searcher import structuredescriptor, searcher, colordescriptor
import cv2
import glob

def buildimgpairs(dataset,colorIndexPath,structureIndexPath,imgfilepath,eximgfilepath,fdbdir=None):
	if fdbdir is None:
		fdbdir = dataset
	with open(imgfilepath, "w") as imgfile:
		with open(eximgfilepath,'w') as eximgfile:
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
				imgfile.write(str(imagePath))
				imgfile.write('\r\n')

				eximgfile.write(str(join(fdbdir,respairs[0]+'.png')))
				eximgfile.write('\r\n')

			eximgfile.close()
	# close index file
	imgfile.close()

if __name__ == '__main__':
	dataset = '/home/zzy/TrainData/MITPlace2Dataset/val_recut_512x680/'
	# dataset = '/home/zzy/work/experiment_image_inpaint/searcher/examples/output/'
	# dataset = "/home/zzy/TrainData/HolidaySet/"
	colorIndexPath = "/home/zzy/work/exi_inpaint/eximage/files/colorfeatures.csv"
	structureIndexPath = "/home/zzy/work/exi_inpaint/eximage/files/strucfeatures.csv"
	imgfile = "/home/zzy/work/exi_inpaint/eximage/files/imgfile.txt"
	eximgfile = "/home/zzy/work/exi_inpaint/eximage/files/eximgfile.txt"
	buildimgpairs(dataset, colorIndexPath, structureIndexPath, imgfile,eximgfile)
	print('build image pairs done.')