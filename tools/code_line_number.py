#coding:utf-8
from __future__ import print_function
import os, re

def getCurrentDir():
	return os.path.abspath(os.path.dirname(__file__))

def getFileExtensions():
	return [
		'.java',
		'.js',
		'.html',
		'.py',
		'.h',
	]

def getSubdirectories():
	root = getCurrentDir()
	return [
		root + '/../point_process',
		root + '/../preprocess',
		root + '/../spatial_temporal',
		root + '/../tools',
	]

def getHorizontalLine():
	return "-------------------------------------------------"

def inDirPatternBlackList(dir_):
	black = [
		'\\btarget\\b',
		'\\.git',
		'\\bthird_party\\b',
	]
	for p in black:
		reg = re.compile(p)
		if reg.search(dir_) is not None:
			return True
	return False

def inFilePatternBlackList(file_):
	black = [
		'jquery',
	]
	for p in black:
		reg = re.compile(p)
		if reg.search(file_) is not None:
			return True
	return False

if __name__ == '__main__':
	for ext in getFileExtensions():
		print(getHorizontalLine())
		print('file extension['+str(ext)+']')
		file_list = {}
		file_count = {}
		line_number = {}
		line_number_exclude_blank = {}
		for dir_ in getSubdirectories():
			dir_ = os.path.abspath(dir_)
			for parent, dirnames, filenames in os.walk(dir_):
				if inDirPatternBlackList(parent) == True:
					continue
				for name in filenames:
					if inFilePatternBlackList(name) == True:
						continue
					if name.endswith(ext):
						file_list[dir_] = file_list.get(dir_,[]) + [os.path.abspath(os.path.join(parent,name))]
						file_count[dir_] = file_count.get(dir_,0) + 1
						with open(os.path.join(parent,name),'rb') as f:
							for line in f:
								line_number[dir_] = line_number.get(dir_,0) + 1
								if len(line.strip()) > 0:
									line_number_exclude_blank[dir_] = line_number_exclude_blank.get(dir_,0) + 1
			print('')
			print('sub directory['+str(dir_)+']')
			print('\tfile_count['+str(file_count.get(dir_,0))+']')
			print('\tline_number['+str(line_number.get(dir_,0))+']')
			print('\tline_number_exclude_blank['+str(line_number_exclude_blank.get(dir_,0))+']')
			# print('\tfile_list['+str(file_list.get(dir_,[]))+']')
		print('')
		total = 0
		for dir_ in file_count:
			total += file_count[dir_]
		print('total_file_count['+str(total)+']')
		total = 0
		for dir_ in line_number:
			total += line_number[dir_]
		print('total_line_number['+str(total)+']')
		total = 0
		for dir_ in line_number_exclude_blank:
			total += line_number_exclude_blank[dir_]
		print('total_line_number_exclude_blank['+str(total)+']')
		print(getHorizontalLine())
