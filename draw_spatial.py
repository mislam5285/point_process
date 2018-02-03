#!/usr/bin/env python
#coding:utf-8

import json
import operator
import os
import sys

import matplotlib
import numpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from spatial import pp_spatial
from spatial import data_loader

import os, sys
root = os.path.abspath(os.path.dirname(__file__))

def draw_spatial_temporal_rnn_contrast_reg(year):
	crime_configs = [
		{
			'DATA_MODE':'real',
			'DATASET':root + '/data/crime2.'+str(year)+'.txt',
			'REG':0.01,
			'will_train':False,
			'will_draw':True,
		},
		{
			'DATA_MODE':'real',
			'DATASET':root + '/data/crime2.'+str(year)+'.txt',
			'REG':0.1,
			'will_train':False,
			'will_draw':True,
		},
		{
			'DATA_MODE':'real',
			'DATASET':root + '/data/crime2.'+str(year)+'.txt',
			'REG':0.2,
			'will_train':False,
			'will_draw':True,
		},
		{
			'DATA_MODE':'real',
			'DATASET':root + '/data/crime2.'+str(year)+'.txt',
			'REG':0.9,
			'will_train':False,
			'will_draw':True,
		},

		{
			'DATA_MODE':'simulate',
			'DATASET':root + '/data/crime2.'+str(year)+'.txt',
			'REG':0.01,
			'will_train':False,
			'will_draw':True,
		},
		{
			'DATA_MODE':'simulate',
			'DATASET':root + '/data/crime2.'+str(year)+'.txt',
			'REG':0.1,
			'will_train':False,
			'will_draw':True,
		},
		{
			'DATA_MODE':'simulate',
			'DATASET':root + '/data/crime2.'+str(year)+'.txt',
			'REG':0.2,
			'will_train':False,
			'will_draw':True,
		},
		{
			'DATA_MODE':'simulate',
			'DATASET':root + '/data/crime2.'+str(year)+'.txt',
			'REG':0.9,
			'will_train':False,
			'will_draw':True,
		},
	]

	for config in crime_configs:
		pp_spatial.DATAMODE = config['DATA_MODE']
		pp_spatial.DATASET = open(config['DATASET'])
		pp_spatial.REG = config['REG']
		pp_spatial.NAME_SCOPE = str(config['DATA_MODE']) + '.' + str(config['REG'])
		pp_spatial.ITERS = 3000
		
		log = root + '/data/crime2.spatial_rnn.contrast.' + str(config['DATA_MODE']) + '.' + str(config['REG']) + '.txt'
		if config['will_train'] == True:
			with open(log,'w') as fw:
				stdout_old = sys.stdout
				sys.stdout = fw
				pp_spatial.run()
				sys.stdout = stdout_old
		
		if config['will_draw'] == True:
			plt.figure()
			curves = [
				{
					'key':'total_loss',
					'label':'Total Loss',
					'color':'red',
				},
				{
					'key':'time_loss',
					'label':'Time Loss',
					'color':'green',
				},
				{
					'key':'mark_loss',
					'label':'Mark Loss',
					'color':'blue',
				},
			]
			data = []
			with open(log) as f:
				for line in f:
					line = eval(line)
					data.append(line)
			
			for curve in curves:
				curve['y'] = map(lambda x:x[curve['key']],data)
				plt.plot(curve['y'],c=curve['color'],lw=1.2,label=curve['label'])
				plt.xticks(fontsize=13)
				plt.yticks(fontsize=13)
				plt.legend(loc='upper right',fontsize=13)

			plt.xlabel('iterations')
			plt.ylabel('Loss')
			plt.title('learning curve for ' + str(config['DATA_MODE']) + ' data, trade off:' + str(config['REG'] ))
			plt.gcf().set_size_inches(5.9, 5., forward=True)

			key = 'crime2_spatial_rnn_contrast_' + str(config['DATA_MODE']) + '_' + str(config['REG']).replace('.','_') + '.png'
			plt.savefig(root + '/pic/%s'%key)

def draw_histogram_time_distribution(year):
	sequences = data_loader.load_crime_samples(open(root + '/data/crime2.'+str(year)+'.txt'))
	sequence = [0.]
	for seq in sequences:
		seq2 = [sequence[-1] + x[0] for x in seq]
		sequence.extend(seq2)
	# with open(root + '/log/spatial.log','w') as fw:
	# 	fw.write(str(sequence))
	curve = {
		'label':'Occur Time Distribution',
		'color':'gray',
		'x_right':sequence[-1] / (24. * 7),
		'y':[]
	}
	week = 0
	event = 0
	while event < len(sequence):
		count = 0
		while event < len(sequence) and sequence[event] < (week + 1) * 7 * 24:
			count += 1
			event += 1
		curve['y'].append(float(count)/len(sequence))
		count = 0
		week += 1
	curve['y'] = curve['y'][-365/7:]
	# print curve['y']
	plt.bar(range(len(curve['y'])),curve['y'],0.8,color=curve['color'],label=curve['label'])
	plt.xticks(fontsize=13)
	plt.yticks(fontsize=13)
	# plt.legend(loc='upper right',fontsize=13)

	plt.xlabel('time (days)')
	plt.ylabel('crime events')
	plt.title('histogram of time distribution for Nov. and Dec. in ' + str(year))
	plt.gcf().set_size_inches(5.9, 5., forward=True)

	key = 'crime2_distribution_time_'+str(year)+'.png'
	plt.savefig(root + '/pic/%s'%key)

def print_predict_contrast_sigma_year(years) :
	will_train_sparial_rnn = False
	will_predict_spatial_rnn = False

	if will_train_sparial_rnn == True:
		for year in years:
			crime_configs = [
				{
					'DATA_MODE':'real',
					'DATASET':root + '/data/crime2.'+str(year)+'.txt',
					'REG':0.01,
				},
				{
					'DATA_MODE':'real',
					'DATASET':root + '/data/crime2.'+str(year)+'.txt',
					'REG':0.1,
				},
				{
					'DATA_MODE':'real',
					'DATASET':root + '/data/crime2.'+str(year)+'.txt',
					'REG':0.2,
				},
				{
					'DATA_MODE':'real',
					'DATASET':root + '/data/crime2.'+str(year)+'.txt',
					'REG':0.9,
				},
			]
			for config in crime_configs:
				pp_spatial.DATAMODE = config['DATA_MODE']
				pp_spatial.DATASET = open(config['DATASET'])
				pp_spatial.REG = config['REG']
				pp_spatial.NAME_SCOPE = str(year) + '.' + str(config['DATA_MODE']) + '.' + str(config['REG'])
				pp_spatial.ITERS = 1000
				
				log_train = root + '/log/train_spatial.log'
				save_target = root + '/log/sess.' + '.' + str(year) + '.' + str(config['DATA_MODE']) + '.' + str(config['REG']) + '.model.log'
				# if config['will_train'] == True:
				with open(log_train,'w') as fw:
					stdout_old = sys.stdout
					sys.stdout = fw
					pp_spatial.run(save_target=save_target)
					sys.stdout = stdout_old
		
	if will_predict_spatial_rnn == True:
		for year in years:
			crime_configs = [
				{
					'DATA_MODE':'real',
					'DATASET':root + '/data/crime2.'+str(year)+'.txt',
					'REG':0.01,
				},
				{
					'DATA_MODE':'real',
					'DATASET':root + '/data/crime2.'+str(year)+'.txt',
					'REG':0.1,
				},
				{
					'DATA_MODE':'real',
					'DATASET':root + '/data/crime2.'+str(year)+'.txt',
					'REG':0.2,
				},
				{
					'DATA_MODE':'real',
					'DATASET':root + '/data/crime2.'+str(year)+'.txt',
					'REG':0.9,
				},
			]
			for config in crime_configs:
				pp_spatial.DATAMODE = config['DATA_MODE']
				pp_spatial.DATASET = open(config['DATASET'])
				pp_spatial.REG = config['REG']
				pp_spatial.NAME_SCOPE = str(year) + '.' +  str(config['DATA_MODE']) + '.' + str(config['REG'])
				pp_spatial.ITERS = 1000
				log_predict = root + '/data/crime2.predict.spatial_rnn.contrast.' + str(year) + '.' + str(config['DATA_MODE']) + '.' + str(config['REG']) + '.txt'
				save_target = root + '/log/sess.' + '.' + str(year) + '.' + str(config['DATA_MODE']) + '.' + str(config['REG']) + '.model.log'
				with open(log_predict,'w') as fw:
					stdout_old = sys.stdout
					sys.stdout = fw
					pp_spatial.run(saved_sess=save_target)
					sys.stdout = stdout_old

if __name__ == '__main__' :

	# draw_spatial_temporal_rnn_contrast_reg(2016)
	# draw_histogram_time_distribution(2009)
	print_predict_contrast_sigma_year([2011,2013,2016])
	pass
	# plt.show()