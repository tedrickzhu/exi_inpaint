# -*- coding: utf-8 -*-
# @Time    : 20-3-3 下午12:39
# @Author  : zhuzhengyi

import argparse
from util.build_img_pairs import buildimgpairs

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='', type=str,help='The dir of images to be recutted.')
	parser.add_argument('--colorf', default='', type=str,help='the dir saved color features of dataset.')
	parser.add_argument('--structf', default='', type=str,help='the dir saved struct features of dataset.')
	parser.add_argument('--resultpath', default='', type=str,help='the filepath of images pairs file to save')
	args = parser.parse_args()

	buildimgpairs(args.dataset, args.colorf, args.structf, args.resultpath)
	print('======buildimgpairs finished======')

