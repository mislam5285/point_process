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

from point_process.hawkes import MHawkes, RPP
from point_process.generator import HawkesGenerator
from point_process.pp_gan import HawkesGAN

from preprocess.screen import Screenor

np.random.seed(137)
os.environ["KERAS_BACKEND"] = "tensorflow"

import os, sys
root = os.path.abspath(os.path.dirname(__file__))

def screen():
	will_screen_paper = False
	will_screen_patent = False
	if will_screen_paper == True:
		screenor = Screenor()
		paper_data = root + '/data/paper3.txt'
		paper_data_raw = root + '/data/paper2.txt'
		result = screenor.screen_paper(paper_data_raw)
		print {'number of paper result':len(result)}
		with open(paper_data,'w') as fw:
			fw.writelines(result)

	if will_screen_patent == True:
		screenor = Screenor()
		patent_data = root + '/data/patent3.txt'
		patent_data_raw = root + '/data/patent2.txt'
		result = screenor.screen_patent(patent_data_raw)
		print {'number of patent result':len(result)}
		with open(patent_data,'w') as fw:
			fw.writelines(result)


def draw_pretrain_learning_generator_convergence(dataset_id, nb_type=1):

	will_train_hawkes = {'1:1':False,'1:5':False,'5:1':False,'3:3':False}
	will_draw = True
	will_draw_mle_curve = {'1:1':True,'1:5':False,'5:1':False,'3:3':False}
	will_draw_val = False

	# preprocess
	dataset_path = root + '/data/' + dataset_id + '.txt'

	to_ratio = lambda x:x.split(':')
	# training
	log_pre_train = {}
	for key in will_train_hawkes:
		log_pre_train[key] = root + '/data/' + dataset_id + '.pretrain.log.' + to_ratio(key)[0] + '.' + to_ratio(key)[1] + '.txt'


	for key in will_train_hawkes:
		if will_train_hawkes[key] == True :
			with open(log_pre_train[key],'w') as f:
				old_stdout = sys.stdout
				sys.stdout = f
				predictor = HawkesGenerator()
				loaded = predictor.load(dataset_path,nb_type=nb_type)
				model = predictor.pre_train(*loaded,max_outer_iter=500/(int(to_ratio(key)[0])+int(to_ratio(key)[1])),
					alpha_iter=int(to_ratio(key)[0]),w_iter=int(to_ratio(key)[1]))
				sys.stdout = old_stdout


	# drawing
	if will_draw == True :
		# plt.figure(figsize=(8,6), dpi=72, facecolor="white")
		# colors = ['red','green','purple']
		keys = [lambda x:x['LL'], lambda x:x['acc'][-1], lambda x:x['mape'][-1]]
		keys_val = [lambda x:x['LL'], lambda x:x['acc_val'][-1], lambda x:x['mape_val'][-1]]
		colors = {'test':'red','val':'blue','early_stop':'green','test_best':'purple'}
		# keys = [lambda x:x['acc'][-1], lambda x:x['mape'][-1]]
		labels_prefix = ['NLL Loss','ACC','MAPE']

		f_pre_train = {}
		nodes_pre_train = {}
		for key in will_draw_mle_curve:
			if will_draw_mle_curve[key] == True:
				f_pre_train[key] = open(log_pre_train[key])
				nodes_pre_train[key] = []


		for key in f_pre_train:
			for line in f_pre_train[key]:
				try:
					node = eval(line)
					nodes_pre_train[key].append(node)
				except:
					print 'error'


		for i in range(3): # loss or acc or mape
			x_left_limit = 0
			x_right_limit = 200

			y_pre_train = {}
			for key in f_pre_train:
				y_pre_train[key] = np.array(
					[float(keys[i](node)) for node in nodes_pre_train[key]])[x_left_limit:x_right_limit+1]

			y_pre_train_val = {}
			for key in f_pre_train:
				y_pre_train_val[key] = np.array(
					[float(keys_val[i](node)) for node in nodes_pre_train[key]])[x_left_limit:x_right_limit+1]

			curves = []
			for key in f_pre_train:
				curve = {
					'rate':key,
					'y_test':y_pre_train[key],
					'y_val':y_pre_train_val[key],
				}
				curves.append(curve)


			for curve in curves: # each curve
				fig = plt.figure()

				# arrange layout
				delta = max(np.max(curve['y_test']),np.max(curve['y_test'])) - min(np.min(curve['y_test']),np.min(curve['y_test']))
				delta /= 30.
				if i == 2: # mape #curve['y_test'][0] > curve['y_test'][-1]:
					y_lower_limit = min(np.min(curve['y_test']),np.min(curve['y_test'])) - delta
					y_upper_limit = 0.15 * np.max(curve['y_test']) + 0.85 * np.min(curve['y_test'])
					y_upper_limit = max(y_upper_limit,np.mean(curve['y_test'][-len(curve['y_test'])/3:]) * 1.5 - 0.5 * y_lower_limit)
				if i == 1:
					y_upper_limit = max(np.max(curve['y_test']),np.max(curve['y_test'])) + delta
					y_lower_limit = 0.8 * np.max(curve['y_test']) + 0.2 * np.min(curve['y_test'])
					y_lower_limit = min(y_lower_limit,np.mean(curve['y_test'][-len(curve['y_test'])/3:]) * 1.5 - 0.5 * y_upper_limit)
				if i == 0:#loss
					y_lower_limit = min(np.min(curve['y_test']),np.min(curve['y_test'])) - delta
					y_upper_limit = 0.8 * np.max(curve['y_test']) + 0.2 * np.min(curve['y_test'])
				# draw curve
				plt.ylim(y_lower_limit, y_upper_limit)
				plt.xlim(x_left_limit,x_right_limit)

				if i == 0:
					plt.plot(np.arange(1,len(curve['y_test'])+1),curve['y_test'],c=colors['test'],lw=1.2,
						label=labels_prefix[i] + ' (on observed.)')
				else:
					plt.plot(np.arange(1,len(curve['y_test'])+1),curve['y_test'],c=colors['test'],lw=1.2,
						label=labels_prefix[i] + ' (on test.)')
				if i == 1:
					j = np.argmax(curve['y_test'])
					plt.plot([j,j],[y_lower_limit,curve['y_test'][j]],':',c=colors['test_best'],lw=1.2,
						label='best ' + labels_prefix[i] + ' (on test.)')
					plt.plot([0,j],[curve['y_test'][j],curve['y_test'][j]],':',c=colors['test_best'],lw=1.2)
				if i == 2:
					j = np.argmin(curve['y_test'])
					plt.plot([j,j],[y_lower_limit,curve['y_test'][j]],':',c=colors['test_best'],lw=1.2,
						label='best ' + labels_prefix[i] + ' (on test.)')
					plt.plot([0,j],[curve['y_test'][j],curve['y_test'][j]],':',c=colors['test_best'],lw=1.2)

				plt.xticks(fontsize=13)
				plt.yticks(fontsize=13,color=colors['test'])
				plt.legend(loc='center right',fontsize=13)

				if i > 0 and will_draw_val == True: # draw another axis
					# arrange layout
					delta = max(np.max(curve['y_val']),np.max(curve['y_val'])) - min(np.min(curve['y_val']),np.min(curve['y_val']))
					delta /= 30.
					if curve['y_val'][0] > curve['y_val'][-1]:
						y_lower_limit = min(np.min(curve['y_val']),np.min(curve['y_val'])) - delta
						y_upper_limit = 0.25 * np.max(curve['y_val']) + 0.75 * np.min(curve['y_val'])
					else:
						y_lower_limit = 0.75 * np.max(curve['y_val']) + 0.25 * np.min(curve['y_val'])
						y_upper_limit = max(np.max(curve['y_val']),np.max(curve['y_val'])) + delta

					# draw curve
					ax = plt.subplot(111).twinx()
					ax.set_ylim(y_lower_limit, y_upper_limit)
					ax.set_xlim(x_left_limit,x_right_limit)

					ax.plot(np.arange(1,len(curve['y_val'])+1),curve['y_val'],'-',c=colors['val'],lw=1.2,
						label=labels_prefix[i] + ' (on val.)')
					if i == 2: # draw early stop
						log_pre_train_early_stop = root + '/data/' + dataset_id + '.pretrain.early_stop.stop_point.json'
						# if will_compute_early_stop == True:
						early_stop = -1
						moving_average = curve['y_val'][0]
						k = 20.
						th = 0.001
						for n in range(1,len(curve['y_val'])):
							av = curve['y_val'][n] * (1./k) + moving_average * (k - 1.) / k
							if moving_average - av <= th:
								early_stop = n
								break
							moving_average = av

						try:
							with open(log_pre_train_early_stop) as fr :
								result = json.load(fr)
								if not result.has_key(dataset_id):
									result[dataset_id] = {}
								if not result[dataset_id].has_key(curve['rate']):
									result[dataset_id][curve['rate']] = {}
								result[dataset_id][curve['rate']]['mape_val'] = {'window':k,'threshold':th,'stop_point':early_stop}
						except:
							result = {dataset_id:{curve['rate']:{'mape_val':{'window':k,'threshold':th,'stop_point':early_stop}}}}
						with open(log_pre_train_early_stop,'w') as fw :
							json.dump(result,fw)

						with open(log_pre_train_early_stop) as fr:
							result = json.load(fr)
							early_stop = result[dataset_id][curve['rate']]['mape_val']['stop_point']

						if early_stop > 0:
							ax.plot([early_stop,early_stop],[y_lower_limit,curve['y_val'][early_stop]],':',c=colors['early_stop'],lw=1.2,
								label='signal of early stop')
							ax.plot([early_stop,x_right_limit+100],[curve['y_val'][early_stop],curve['y_val'][early_stop]],':',
								c=colors['early_stop'],lw=1.2)
					plt.xticks(fontsize=13)
					plt.yticks(fontsize=13,color=colors['val'])
					# plt.legend()
					plt.legend(loc='upper right',fontsize=13) #bbox_to_anchor=(0.31,0.8)
					# plt.legend(fontsize=13)
					# plt.gca().add_artist(legend_test)

				plt.xlabel('iterations')
				plt.title('learning curve for ' + labels_prefix[i] + ' ($n_{{em}}:n_{{grad}}$=' + curve['rate'] + ')')
				plt.gcf().set_size_inches(5.9, 5., forward=True)

				if i == 0: key = '' + dataset_id + '_gan_pretrain_learning_NLL_' + to_ratio(key)[0] + '_' + to_ratio(key)[1] +'.png'
				if i == 1: key = '' + dataset_id + '_gan_pretrain_learning_ACC_' + to_ratio(key)[0] + '_' + to_ratio(key)[1] +'.png'
				if i == 2: key = '' + dataset_id + '_gan_pretrain_learning_MAPE_' + to_ratio(key)[0] + '_' + to_ratio(key)[1] +'.png'
				if i == 0: plt.yticks(fontsize=11)
				plt.savefig(root + '/pic/%s'%key)


def draw_full_train_learning_gan_convergence(dataset_id, nb_type=1):
	will_pretrain_with_mle = False

	will_train_mse_noise_dropout = False
	will_train_wgan_noise_dropout = False

	will_train_mse_noise_sample = False
	will_train_wgan_noise_sample = False

	will_train_mse_with_wgan_noise_sample = False
	will_train_mae_with_wgan_noise_sample = False
	will_draw = True
	will_draw_curve = {
			'mle_only':True,
			'mse_noise':False,
			'wgan_noise':False,
			'mse_noise_sample':False,
			'wgan_noise_sample':True,
			'mse_with_wgan_noise_sample':True,
			'mae_with_wgan_noise_sample':True,
		}
	will_draw_val = True
	will_draw_val_curve = {
			'mle_only':True,
			'mse_noise':False,
			'wgan_noise':False,
			'mse_noise_sample':False,
			'wgan_noise_sample':True,
			'mse_with_wgan_noise_sample':True,
			'mae_with_wgan_noise_sample':True,
		}

	dataset_path = root + '/data/' + dataset_id + '.txt'

	# pre-training
	ratio = [1,1]
	ratio_key = str(ratio[0]) + ':' + str(ratio[1])
	log_mle_only = root + '/data/' + dataset_id + '.pretrain.log.' + str(ratio[0]) + '.' + str(ratio[1]) + '.txt'

	if will_pretrain_with_mle == True :
		with open(log_mle_only,'w') as f:
			old_stdout = sys.stdout
			sys.stdout = f
			predictor = HawkesGenerator()
			loaded = predictor.load(dataset_path,nb_type=nb_type)
			model = predictor.pre_train(*loaded,max_outer_iter=500/(int(ratio[0])+int(ratio[1])),
				alpha_iter=int(ratio[0]),w_iter=int(ratio[1]))
			sys.stdout = old_stdout


	log_pre_train_early_stop = root + '/data/' + dataset_id + '.pretrain.early_stop.stop_point.json'
	with open(log_pre_train_early_stop) as fr:
		result = json.load(fr)
		early_stop = result[dataset_id][ratio_key]['mape_val']['stop_point']

	full_train_start = early_stop
	assert full_train_start > 0


	log_mse_noise = root + '/data/' + dataset_id + '.fulltrain.mse_noise.log.txt'
	if will_train_mse_noise_dropout == True:
		mse_weight = 1.
		gan_weight = 0.
		stddev = 15.
		with open(log_mse_noise,'w') as f:
			old_stdout = sys.stdout
			sys.stdout = f
			gan = HawkesGAN()
			try:
				gan.gen.sequence_weights = json.load(
					open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json'))
			except:
				loaded = gan.gen.load(dataset_path,nb_type=nb_type)
				sys.stdout = open(root + '/log/pretrain.log','w')
				gan.gen.pre_train(*loaded,early_stop=full_train_start,max_outer_iter=500/(int(ratio[0])+int(ratio[1])),
					alpha_iter=int(ratio[0]),w_iter=int(ratio[1]))
				sys.stdout = f
				with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json','w') as fw:
					json.dump(gan.gen.sequence_weights,fw)
			# exit()
			loaded = gan.load(dataset_path,nb_type=nb_type)
			gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True,stddev=stddev)
			sys.stdout = old_stdout

	log_wgan_noise = root + '/data/' + dataset_id + '.fulltrain.wgan_noise.log.txt'
	if will_train_wgan_noise_dropout == True:
		mse_weight = 0.
		gan_weight = 1.
		stddev = 15.
		wgan_clip = 2.
		with open(log_wgan_noise,'w') as f:
			old_stdout = sys.stdout
			sys.stdout = f
			gan = HawkesGAN()
			try:
				gan.gen.sequence_weights = json.load(
					open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json'))
			except:
				loaded = gan.gen.load(dataset_path,nb_type=nb_type)
				sys.stdout = open(root + '/log/pretrain.log','w')
				gan.gen.pre_train(*loaded,early_stop=full_train_start,max_outer_iter=500/(int(ratio[0])+int(ratio[1])),
					alpha_iter=int(ratio[0]),w_iter=int(ratio[1]))
				sys.stdout = f
				with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json','w') as fw:
					json.dump(gan.gen.sequence_weights,fw)
			# exit()
			loaded = gan.load(dataset_path,nb_type=nb_type)
			gan.full_train(*loaded,train_gan_method='wgan',max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True,
				stddev=stddev,wgan_clip=wgan_clip)
			sys.stdout = old_stdout


	log_mse_noise_sample = root + '/data/' + dataset_id + '.fulltrain.mse_noise_sample.log.txt'
	if will_train_mse_noise_sample == True:
		mse_weight = 1.
		gan_weight = 0.
		stddev = 15.
		sample_stddev = 15.
		with open(log_mse_noise_sample,'w') as f:
			old_stdout = sys.stdout
			sys.stdout = f
			gan = HawkesGAN()
			try:
				gan.gen.sequence_weights = json.load(
					open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json'))
			except:
				loaded = gan.gen.load(dataset_path,nb_type=nb_type)
				sys.stdout = open(root + '/log/pretrain.log','w')
				gan.gen.pre_train(*loaded,early_stop=full_train_start,max_outer_iter=500/(int(ratio[0])+int(ratio[1])),
					alpha_iter=int(ratio[0]),w_iter=int(ratio[1]))
				sys.stdout = f
				with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json','w') as fw:
					json.dump(gan.gen.sequence_weights,fw)
			# exit()
			loaded = gan.load(dataset_path,nb_type=nb_type)
			gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True,
				stddev=stddev,sample_stddev=sample_stddev)
			sys.stdout = old_stdout

	log_wgan_noise_sample = root + '/data/' + dataset_id + '.fulltrain.wgan_noise_sample.log.txt'
	if will_train_wgan_noise_sample == True:
		mse_weight = 0.
		gan_weight = 1.
		stddev = 10.
		sample_stddev = 10.
		wgan_clip = 2.
		with open(log_wgan_noise_sample,'w') as f:
			old_stdout = sys.stdout
			sys.stdout = f
			gan = HawkesGAN()
			try:
				gan.gen.sequence_weights = json.load(
					open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json'))
			except:
				loaded = gan.gen.load(dataset_path,nb_type=nb_type)
				sys.stdout = open(root + '/log/pretrain.log','w')
				gan.gen.pre_train(*loaded,early_stop=full_train_start,max_outer_iter=500/(int(ratio[0])+int(ratio[1])),
					alpha_iter=int(ratio[0]),w_iter=int(ratio[1]))
				sys.stdout = f
				with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json','w') as fw:
					json.dump(gan.gen.sequence_weights,fw)
			# exit()
			loaded = gan.load(dataset_path,nb_type=nb_type)
			gan.full_train(*loaded,train_gan_method='wgan',max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True,
				stddev=stddev,sample_stddev=sample_stddev,wgan_clip=wgan_clip)
			sys.stdout = old_stdout

	log_mse_with_wgan_noise_sample = root + '/data/' + dataset_id + '.fulltrain.mse_wgan.noise_sample.log.txt'
	if will_train_mse_with_wgan_noise_sample == True:
		mse_weight = 0.5
		gan_weight = 0.5
		stddev = 15.
		sample_stddev = 15.
		wgan_clip = 2.
		with open(log_mse_with_wgan_noise_sample,'w') as f:
			old_stdout = sys.stdout
			sys.stdout = f
			gan = HawkesGAN()
			try:
				gan.gen.sequence_weights = json.load(
					open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json'))
			except:
				loaded = gan.gen.load(dataset_path,nb_type=nb_type)
				sys.stdout = open(root + '/log/pretrain.log','w')
				gan.gen.pre_train(*loaded,early_stop=full_train_start,max_outer_iter=500/(int(ratio[0])+int(ratio[1])),
					alpha_iter=int(ratio[0]),w_iter=int(ratio[1]))
				sys.stdout = f
				with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json','w') as fw:
					json.dump(gan.gen.sequence_weights,fw)
			# exit()
			loaded = gan.load(dataset_path,nb_type=nb_type)
			gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True,
				stddev=stddev,sample_stddev=sample_stddev,wgan_clip=wgan_clip)
			sys.stdout = old_stdout

	log_mae_with_wgan_noise_sample = root + '/data/' + dataset_id + '.fulltrain.mae_wgan.noise_sample.log.txt'
	if will_train_mae_with_wgan_noise_sample == True:
		mse_weight = 0.5
		gan_weight = 0.5
		stddev = 15.
		sample_stddev = 15.
		wgan_clip = 2.
		with open(log_mae_with_wgan_noise_sample,'w') as f:
			old_stdout = sys.stdout
			sys.stdout = f
			gan = HawkesGAN()
			try:
				gan.gen.sequence_weights = json.load(
					open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json'))
			except:
				loaded = gan.gen.load(dataset_path,nb_type=nb_type)
				sys.stdout = open(root + '/log/pretrain.log','w')
				gan.gen.pre_train(*loaded,early_stop=full_train_start,max_outer_iter=500/(int(ratio[0])+int(ratio[1])),
					alpha_iter=int(ratio[0]),w_iter=int(ratio[1]))
				sys.stdout = f
				with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json','w') as fw:
					json.dump(gan.gen.sequence_weights,fw)
			# exit()
			loaded = gan.load(dataset_path,nb_type=nb_type)
			gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True,
				stddev=stddev,sample_stddev=sample_stddev,wgan_clip=wgan_clip,hawkes_output_loss='mae')
			sys.stdout = old_stdout

	# drawing
	if will_draw == True :
		# plt.figure(figsize=(8,6), dpi=72, facecolor="white")
		colors = {
			'mle_only':'red',
			'mse_noise':'blue',
			'wgan_noise':'green',
			'mse_noise_sample':'purple',
			'wgan_noise_sample':'orange',
			'mse_with_wgan_noise_sample':'blue',
			'mae_with_wgan_noise_sample':'purple',
			'early_stop':'green',
		}
		keys = [lambda x:x['acc'][-1], lambda x:x['mape'][-1]]
		keys_val = [lambda x:x['acc_val'][-1], lambda x:x['mape_val'][-1]]
		labels_prefix = ['ACC','MAPE']
		labels_suffix = {
			'mle_only':'MLE pretrain',
			'mse_noise':'ppGAN 1',
			'wgan_noise':'ppGAN 2',
			'mse_noise_sample':'ppGAN 3',
			'wgan_noise_sample':'MSE',
			'mse_with_wgan_noise_sample':'MSE+WGAN',
			'mae_with_wgan_noise_sample':'MAE+WGAN',
		}
		log_full_train = {
			'mle_only':log_mle_only,
			'mse_noise':log_mse_noise,
			'wgan_noise':log_wgan_noise,
			'mse_noise_sample':log_mse_noise_sample,
			'wgan_noise_sample':log_wgan_noise_sample,
			'mse_with_wgan_noise_sample':log_mse_with_wgan_noise_sample,
			'mae_with_wgan_noise_sample':log_mae_with_wgan_noise_sample,
		}

		f_full_train = {}
		nodes_full_train = {}
		for curve_key in will_draw_curve:
			if will_draw_curve[curve_key] == True:
				f_full_train[curve_key] = open(log_full_train[curve_key])
				nodes_full_train[curve_key] = []


		for curve_key in f_full_train:
			for line in f_full_train[curve_key]:
				try:
					node = eval(line)
					nodes_full_train[curve_key].append(node)
				except:
					print 'error'


		for i in range(len(keys)):
			plt.figure()

			# arrange layout
			y_full_train = {}
			for curve_key in nodes_full_train:
				y_full_train[curve_key] = np.array([float(keys[i](node)) for node in nodes_full_train[curve_key]])


			true_upper = 0.
			true_lower = min(y_full_train['mle_only'])
			for curve_key in y_full_train:
				max_ = max(y_full_train[curve_key])
				min_ = min(y_full_train[curve_key])
				if max_ > true_upper: true_upper = max_
				if min_ < true_lower: true_lower = min_ 
			delta = true_upper - true_lower
			delta /= 30.
			x_left_limit = 0
			x_right_limit = 420
			if y_full_train['mle_only'][0] > y_full_train['mle_only'][-1]:
				y_lower_limit = true_lower - delta
				y_upper_limit = 0.15 * true_upper + 0.85 * true_lower
			else:
				y_lower_limit = 0.85 * true_upper + 0.15 * true_lower
				y_upper_limit = true_upper + delta

			plt.ylim(y_lower_limit, y_upper_limit)
			plt.xlim(0,x_right_limit)

			# draw curve
			for curve_key in y_full_train:
				if curve_key == 'mle_only':
					plt.plot(np.arange(1,len(y_full_train['mle_only'])+1),y_full_train['mle_only'],c=colors['mle_only'],lw=1.2,
						label=labels_suffix['mle_only'] + ' (on test.)')
				else:
					plt.plot(np.arange(full_train_start,len(y_full_train[curve_key])+full_train_start),y_full_train[curve_key],c=colors[curve_key],lw=1.2,
						label=labels_suffix[curve_key] + ' (on test.)')

			plt.legend(loc='upper right')#,fontsize=13)
			if i == 0: plt.legend(loc='lower right')#,fontsize=13)
			plt.ylabel(labels_prefix[i] + ' as of the last year of test.')
			plt.xlabel('iterations')


			if will_draw_val == True: # draw another axis
				# arrange layout
				y_full_train_val = {}
				for curve_key in nodes_full_train:
					if will_draw_val_curve[curve_key] == True:
						y_full_train_val[curve_key] = np.array([float(keys_val[i](node)) for node in nodes_full_train[curve_key]])

				true_upper = 0.
				true_lower = min(y_full_train_val['mle_only'])
				for curve_key in y_full_train_val:
					max_ = max(y_full_train_val[curve_key])
					min_ = min(y_full_train_val[curve_key])
					if max_ > true_upper: true_upper = max_
					if min_ < true_lower: true_lower = min_ 
				delta = true_upper - true_lower
				delta /= 30.
				x_left_limit = 0
				x_right_limit = 420
				if y_full_train_val['mle_only'][0] > y_full_train_val['mle_only'][-1]:
					y_lower_limit = true_lower - delta
					y_upper_limit = 0.15 * true_upper + 0.85 * true_lower
				else:
					y_lower_limit = 0.85 * true_upper + 0.15 * true_lower
					y_upper_limit = true_upper + delta


				ax = plt.subplot(111).twinx()
				ax.set_ylim(y_lower_limit, y_upper_limit)
				ax.set_xlim(x_left_limit,x_right_limit)

				# draw curve
				for curve_key in y_full_train_val:
					if curve_key == 'mle_only':
						ax.plot(np.arange(1,len(y_full_train_val['mle_only'])+1),y_full_train_val['mle_only'],'--',c=colors['mle_only'],lw=1.2,
							label=labels_suffix['mle_only'] + ' (on val.)')
					else:
						ax.plot(np.arange(full_train_start,len(y_full_train_val[curve_key])+full_train_start),y_full_train_val[curve_key],'--',c=colors[curve_key],lw=1.2,
							label=labels_suffix[curve_key] + ' (on val.)')
					#print y_full_train_val[curve_key][-1]
					if early_stop > 0 and curve_key == 'mle_only':
						ax.plot([early_stop,early_stop],[y_lower_limit,y_upper_limit],':',c=colors['early_stop'],lw=1.2,
							label='signal of early stop')
						ax.plot([early_stop,x_right_limit+100],[y_full_train_val[curve_key][early_stop],y_full_train_val[curve_key][early_stop]],':',
							c=colors['early_stop'],lw=1.2)
				# plt.xticks(fontsize=13)
				# plt.yticks(fontsize=13,color='red')#colors['val'])
				# plt.legend()
				plt.legend(loc='lower center')#,fontsize=13) #bbox_to_anchor=(0.31,0.8)
				# plt.legend(fontsize=13)
				# plt.gca().add_artist(legend_test)


			plt.title('learning curve for ' + labels_prefix[i])
			plt.ylabel(labels_prefix[i] + ' as of the last year of val.')
			plt.legend(loc='upper right')
			if i == 1: plt.legend(loc='center right')
			plt.gcf().set_size_inches(5.9, 5., forward=True)

			#plt.show()
			if i == 0: key = '' + dataset_id + '_gan_fulltrain_learning_mle_gan_test_ACC.png'
			if i == 1: key = '' + dataset_id + '_gan_fulltrain_learning_mle_gan_test_MAPE.png'
			# plt.xticks(fontsize=13)
			# plt.yticks(fontsize=13)
			# plt.legend(fontsize=13)
			plt.savefig(root + '/pic/%s'%key)




# def draw_full_train_learning_discriminative_convergence(dataset_id, nb_type=1): # mse mae mape ... noise/dropout survey ...
#     will_pretrain_with_mle = False
#     will_train_mle_to_mse = False

#     will_train_mse_only = False
#     will_train_mse_only_noise = False
#     will_train_mse_noise_dropout = False
#     will_train_mle_mae = False

#     will_draw = True
#     # preprocess
#     dataset_path = root + '/data/' + dataset_id + '.txt'

#     # pre-training
#     ratio = [1,1]
#     log_mle_only = root + '/data/' + dataset_id + '.pretrain.log.' + str(ratio[0]) + '.' + str(ratio[1]) + '.txt'

#     if will_pretrain_with_mle == True :
#         with open(log_mle_only,'w') as f:
#             old_stdout = sys.stdout
#             sys.stdout = f
#             predictor = HawkesGenerator()
#             loaded = predictor.load(dataset_path,nb_type=nb_type)
#             model = predictor.pre_train(*loaded,max_outer_iter=10)
#             sys.stdout = old_stdout

#     # full-training
#     log_mle_to_mse = root + '/data/' + dataset_id + '.fulltrain.mse.log.txt'

#     alpha_iter = ratio[0]
#     w_iter = ratio[1]

#     log_pre_train_early_stop = root + '/data/' + dataset_id + '.pretrain.early_stop.stop_point.json'
#     with open(log_pre_train_early_stop) as fr:
#         result = json.load(fr)
#         ratio_key = str(ratio[0]) + ':' + str(ratio[1])
#         early_stop = result[dataset_id][ratio_key]['mape_val']['stop_point']

#     full_train_start = early_stop
#     assert full_train_start > 0

#     if will_train_mle_to_mse == True :
#         mse_weight = 1.
#         gan_weight = 0.
#         with open(log_mle_to_mse,'w') as f:
#             old_stdout = sys.stdout
#             sys.stdout = f
#             gan = HawkesGAN()
#             try:
#                 gan.gen.sequence_weights = json.load(
#                     open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json'))
#             except:
#                 loaded = gan.gen.load(dataset_path,nb_type=nb_type)
#                 sys.stdout = open(root + '/log/pretrain.log','w')
#                 gan.gen.pre_train(*loaded,early_stop=full_train_start,max_outer_iter=500/(int(ratio[0])+int(ratio[1])),
#                     alpha_iter=int(ratio[0]),w_iter=int(ratio[1]))
#                 sys.stdout = f
#                 with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json','w') as fw:
#                     json.dump(gan.gen.sequence_weights,fw)
#             # exit()
#             loaded = gan.load(dataset_path,nb_type=nb_type)
#             gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight)
#             sys.stdout = old_stdout

#     log_mse_only = root + '/data/' + dataset_id + '.fulltrain.mse_only.log.txt'
#     if will_train_mse_only == True:
#         mse_weight = 1.
#         gan_weight = 0.
#         with open(log_mse_only,'w') as f:
#             old_stdout = sys.stdout
#             sys.stdout = f
#             gan = HawkesGAN()
#             gan.gen.sequence_weights = None
#             # exit()
#             loaded = gan.load(dataset_path,nb_type=nb_type)
#             gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_pretrain=False)
#             sys.stdout = old_stdout

#     log_mse_only_noise = root + '/data/' + dataset_id + '.fulltrain.mse_only_noise.log.txt'
#     if will_train_mse_only_noise == True:
#         mse_weight = 1.
#         gan_weight = 0.
#         with open(log_mse_only_noise,'w') as f:
#             old_stdout = sys.stdout
#             sys.stdout = f
#             gan = HawkesGAN()
#             gan.gen.sequence_weights = None
#             # exit()
#             loaded = gan.load(dataset_path,nb_type=nb_type)
#             gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_pretrain=False,need_noise_dropout=True)
#             sys.stdout = old_stdout

#     log_mse_noise = root + '/data/' + dataset_id + '.fulltrain.mse_noise.log.txt'
#     # if will_train_mse_noise_dropout == True:
#     #     mse_weight = 1.
#     #     gan_weight = 0.
#     #     with open(log_mse_noise,'w') as f:
#     #         old_stdout = sys.stdout
#     #         sys.stdout = f
#     #         gan = HawkesGAN()
#     #         try:
#     #             gan.gen.sequence_weights = json.load(
#     #                 open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json'))
#     #         except:
#     #             loaded = gan.gen.load(dataset_path,nb_type=nb_type)
#     #             sys.stdout = open(root + '/log/pretrain.log','w')
#     #             gan.gen.pre_train(*loaded,early_stop=full_train_start,max_outer_iter=500/(int(ratio[0])+int(ratio[1])),
#                     # alpha_iter=int(ratio[0]),w_iter=int(ratio[1]))
#     #             sys.stdout = f
#     #             with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json','w') as fw:
#     #                 json.dump(gan.gen.sequence_weights,fw)
#     #         # exit()
#     #         loaded = gan.load(dataset_path,nb_type=nb_type)
#     #         gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True)
#     #         sys.stdout = old_stdout

#     log_mle_to_mae = root + '/data/' + dataset_id + '.fulltrain.mae.log.txt'
#     if will_train_mle_mae == True:
#         mse_weight = 1.
#         gan_weight = 0.
#         with open(log_mle_to_mae,'w') as f:
#             old_stdout = sys.stdout
#             sys.stdout = f
#             gan = HawkesGAN()
#             try:
#                 gan.gen.sequence_weights = json.load(
#                     open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json'))
#             except:
#                 loaded = gan.gen.load(dataset_path,nb_type=nb_type)
#                 sys.stdout = open(root + '/log/pretrain.log','w')
#                 gan.gen.pre_train(*loaded,early_stop=full_train_start,max_outer_iter=500/(int(ratio[0])+int(ratio[1])),
#                     alpha_iter=int(ratio[0]),w_iter=int(ratio[1]))
#                 sys.stdout = f
#                 with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.json','w') as fw:
#                     json.dump(gan.gen.sequence_weights,fw)
#             # exit()
#             loaded = gan.load(dataset_path,nb_type=nb_type)
#             gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=False,hawkes_output_loss='mae')
#             sys.stdout = old_stdout


#     # drawing
#     if will_draw == True :
#         # plt.figure(figsize=(8,6), dpi=72, facecolor="white")
#         colors = {'mle_only':'red','mle_mse':'green','mle_mae':'orange','mse_only':'blue','mse_noise':'purple'}
#         keys = [lambda x:x['acc'][-1], lambda x:x['mape'][-1]]
#         labels_prefix = ['ACC','MAPE']
#         labels_suffix = {
#             'mle_only':'MLE only',
#             'mle_mse':'MSE without noise',
#             'mle_mae':'MAE without noise',
#             'mse_only':'MSE only',# with optimal $\\beta$',
#             'mse_noise':'MSE after MLE',
#         }

#         f_mle_only = open(log_mle_only)
#         f_mle_to_mse = open(log_mle_to_mse)
#         f_mle_to_mae = open(log_mle_to_mae)
#         f_mse_only = open(log_mse_only)
#         f_mse_only_noise = open(log_mse_only_noise)
#         f_mse_noise = open(log_mse_noise)
#         nodes_mle_only = []
#         nodes_mle_to_mse = []
#         nodes_mle_to_mae = []
#         nodes_mse_only = []
#         nodes_mse_only_noise = []
#         nodes_mse_noise = []

#         for line in f_mle_only:
#             try:
#                 node = eval(line)
#                 nodes_mle_only.append(node)
#             except:
#                 print 'error'

#         for line in f_mle_to_mse:
#             try:
#                 node = eval(line)
#                 nodes_mle_to_mse.append(node)
#             except:
#                 print 'error'

#         for line in f_mle_to_mae:
#             try:
#                 node = eval(line)
#                 nodes_mle_to_mae.append(node)
#             except:
#                 print 'error'

#         for line in f_mse_only:
#             try:
#                 node = eval(line)
#                 nodes_mse_only.append(node)
#             except:
#                 print 'error'

#         for line in f_mse_only_noise:
#             try:
#                 node = eval(line)
#                 nodes_mse_only_noise.append(node)
#             except:
#                 print 'error'

#         for line in f_mse_noise:
#             try:
#                 node = eval(line)
#                 nodes_mse_noise.append(node)
#             except:
#                 print 'error'

#         for i in range(len(keys)):
#             plt.figure()

#             # arrange layout
#             y_mle_only = np.array([float(keys[i](node)) for node in nodes_mle_only])
#             y_mle_to_mse = np.array([float(keys[i](node)) for node in nodes_mle_to_mse])
#             y_mle_to_mae = np.array([float(keys[i](node)) for node in nodes_mle_to_mae])
#             y_mse_only = np.array([float(keys[i](node)) for node in nodes_mse_only])
#             y_mse_only_noise = np.array([float(keys[i](node)) for node in nodes_mse_only_noise])
#             y_mse_noise = np.array([float(keys[i](node)) for node in nodes_mse_noise])

#             delta = max(np.max(y_mle_only),np.max(y_mle_to_mse)) - min(np.min(y_mle_only),np.min(y_mle_to_mse))
#             delta /= 30.
#             x_left_limit = 0
#             x_right_limit = 420
#             if y_mle_only[0] > y_mle_only[-1]:
#                 y_lower_limit = min(np.min(y_mle_only),np.min(y_mse_noise)) - delta
#                 y_upper_limit = 0.15 * np.max(y_mle_only) + 0.85 * np.min(y_mle_only)
#             else:
#                 y_lower_limit = 0.85 * np.max(y_mle_only) + 0.15 * np.min(y_mle_only)
#                 y_upper_limit = max(np.max(y_mle_only),np.max(y_mse_noise)) + delta

#             plt.ylim(y_lower_limit, y_upper_limit)
#             plt.xlim(0,x_right_limit)

#             # draw curve
#             plt.plot(np.arange(1,len(y_mle_only)+1),y_mle_only,c=colors['mle_only'],lw=1.2,
#                 label=labels_suffix['mle_only'])
#             plt.plot(np.arange(0,len(y_mse_only)+0),y_mse_only,c=colors['mse_only'],lw=1.2,
#                 label=labels_suffix['mse_only'])
#             plt.plot(np.arange(full_train_start,len(y_mse_noise)+full_train_start),y_mse_noise,c=colors['mse_noise'],lw=1.2,
#                 label=labels_suffix['mse_noise'])
#             plt.plot(np.arange(full_train_start,len(y_mle_to_mse)+full_train_start),y_mle_to_mse,c=colors['mle_mse'],lw=1.2,
#                 label=labels_suffix['mle_mse'])
#             plt.plot(np.arange(full_train_start,len(y_mle_to_mae)+full_train_start),y_mle_to_mae,c=colors['mle_mae'],lw=1.2,
#                 label=labels_suffix['mle_mae'])


#             plt.xlabel('iterations')
#             plt.title('learning curve for ' + labels_prefix[i])
#             plt.legend(loc='upper right')
#             plt.gcf().set_size_inches(5.9, 5., forward=True)

#             #plt.show()
#             if i == 0: key = '' + dataset_id + '_gan_fulltrain_learning_mle_mse_test_ACC.png'
#             if i == 1: key = '' + dataset_id + '_gan_fulltrain_learning_mle_mse_test_MAPE.png'
#             plt.xticks(fontsize=13)
#             plt.yticks(fontsize=13)
#             plt.legend(fontsize=13)
#             plt.savefig(root + '/pic/%s'%key)

		
def draw_fix_train_non_self_m_hawkes(dataset_id, nb_type=1):
	will_train_rpp = False
	will_train_hawkes = False
	will_draw = True
	# preprocess
	dataset_path = root + '/data/' + dataset_id + '.txt'
	
	# training
	mape_acc_data = root + '/data/' + dataset_id + '.hawkes.mape_acc.json'
	if will_train_rpp == True :
		predictor = RPP()
		loaded = predictor.load(dataset_path,cut=10,nb_type=nb_type)
		rpp_10_result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
		with open(mape_acc_data) as fr :
			result = json.load(fr)
			result['self_nonself']['mape']['proposed-total'] = rpp_10_result['av_mape']
			result['self_nonself']['mape']['proposed-self'] = rpp_10_result['av_mape_self']
			result['self_nonself']['mape']['proposed-nonself'] = rpp_10_result['av_mape_nonself']
			result['self_nonself']['acc']['proposed-total'] = rpp_10_result['av_acc']
			result['self_nonself']['acc']['proposed-self'] = rpp_10_result['av_acc_self']
			result['self_nonself']['acc']['proposed-nonself'] = rpp_10_result['av_acc_nonself']
		with open(mape_acc_data,'w') as fw :
			json.dump(result,fw)

	if will_train_hawkes == True :
		predictor = MHawkes()
		loaded = predictor.load(dataset_path,cut=10,nb_type=nb_type)
		hawkes_10_result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
		with open(mape_acc_data) as fr :
			result = json.load(fr)
			result['self_nonself']['mape']['hawkes-total'] = hawkes_10_result['av_mape']
			result['self_nonself']['mape']['hawkes-self'] = hawkes_10_result['av_mape_self']
			result['self_nonself']['mape']['hawkes-nonself'] = hawkes_10_result['av_mape_nonself']
			result['self_nonself']['acc']['hawkes-total'] = hawkes_10_result['av_acc']
			result['self_nonself']['acc']['hawkes-self'] = hawkes_10_result['av_acc_self']
			result['self_nonself']['acc']['hawkes-nonself'] = hawkes_10_result['av_acc_nonself']
		with open(mape_acc_data,'w') as fw :
			json.dump(result,fw)

	# drawing
	if will_draw == True :
		with open(mape_acc_data) as f:
			graph = json.load(f)
			subgraphs = [graph['self_nonself']['mape'],graph['self_nonself']['acc']]
		colors = ['red','blue','green']
		titles = ['total','non-self','self']
		keys = [
			['proposed-total','proposed-nonself','proposed-self'],
			['hawkes-total','hawkes-nonself','hawkes-self'],
		]
		line_type = ['-','--']
		model_name = ['Proposed','m-hawks']

		for i in [0,1] : # mape graph or acc graph
			plt.figure()
			for j in range(len(titles)): # total or self or nonself
				for k in [0,1] : # proposed or hawkes
					_curve = subgraphs[i][keys[k][j]]
					y = np.array([float(e) for e in _curve])
					plt.plot(np.arange(1,len(y)+1),y,line_type[k],c=colors[j],lw=1.2,label=model_name[k] + "(" + titles[j] + ")")
					plt.scatter(np.arange(1,len(y)+1),y,c=colors[j],lw=0)
			if i == 0:
				plt.xlabel('')
				plt.ylabel('')
				plt.legend(loc='upper left')
				plt.title('MAPE(10 years of training)')
			if i == 1:
				plt.xlabel('')
				plt.ylabel('')
				plt.legend(loc='lower left')
				plt.title('ACC(10 years of training)')

			plt.gcf().set_size_inches(5.9, 5., forward=True)
			# plt.show()
			if i == 0: key = '' + dataset_id + '_fixtrain_non_self_m_hawks_mape.png'
			if i == 1: key = '' + dataset_id + '_fixtrain_non_self_m_hawks_acc.png'
			# plt.xticks(fontsize=13)
			# plt.yticks(fontsize=13)
			# plt.legend(fontsize=13)
			plt.savefig(root + '/pic/%s'%key)


def draw_full_train_contrast_mape_acc(dataset_id, nb_type=1):
	will_draw = True
	will_draw_curve = {
		'mle_only':True,
		'mse_noise':False,
		'wgan_noise':False,
		'mse_noise_sample':False,
		'wgan_noise_sample':True,
		'mse_with_wgan_noise_sample':True,
		'mae_with_wgan_noise_sample':True,
	}
	# preprocess
	dataset_path = root + '/data/' + dataset_id + '.txt'

	# pre-training
	ratio = [1,1]
	log_mle_only = root + '/data/' + dataset_id + '.pretrain.log.' + str(ratio[0]) + '.' + str(ratio[1]) + '.txt'

	# full-training
	# log_mle_to_mse = root + '/data/' + dataset_id + '.fulltrain.mse.log.txt'

	log_pre_train_early_stop = root + '/data/' + dataset_id + '.pretrain.early_stop.stop_point.json'
	with open(log_pre_train_early_stop) as fr:
		result = json.load(fr)
		ratio_key = str(ratio[0]) + ':' + str(ratio[1])
		early_stop = result[dataset_id][ratio_key]['mape_val']['stop_point']

	full_train_start = early_stop
	assert full_train_start > 0

	log_mse_noise = root + '/data/' + dataset_id + '.fulltrain.mse_noise.log.txt'
	log_wgan_noise = root + '/data/' + dataset_id + '.fulltrain.wgan_noise.log.txt'
	log_mse_noise_sample = root + '/data/' + dataset_id + '.fulltrain.mse_noise_sample.log.txt'
	log_wgan_noise_sample = root + '/data/' + dataset_id + '.fulltrain.wgan_noise_sample.log.txt'
	log_mse_with_wgan_noise_sample = root + '/data/' + dataset_id + '.fulltrain.mse_wgan.noise_sample.log.txt'
	log_mae_with_wgan_noise_sample = root + '/data/' + dataset_id + '.fulltrain.mae_wgan.noise_sample.log.txt'


	# drawing
	if will_draw == True :
		# plt.figure(figsize=(8,6), dpi=72, facecolor="white")
		colors = {
			'mle_only':'red',
			'mse_noise':'blue',
			'wgan_noise':'green',
			'mse_noise_sample':'purple',
			'wgan_noise_sample':'orange',
			'mse_with_wgan_noise_sample':'blue',
			'mae_with_wgan_noise_sample':'purple',
			'early_stop':'green',
		}
		keys = ['acc','mape']
		labels_prefix = ['ACC','MAPE']
		legend_loc = ['upper right','lower right']
		labels_suffix = {
			'mle_only':'MLE pretrain',
			'mse_noise':'ppGAN 1',
			'wgan_noise':'ppGAN 2',
			'mse_noise_sample':'ppGAN 3',
			'wgan_noise_sample':'MSE',
			'mse_with_wgan_noise_sample':'MSE+WGAN',
			'mae_with_wgan_noise_sample':'MAE+WGAN',
		}
		log_full_train = {
			'mle_only':log_mle_only,
			'mse_noise':log_mse_noise,
			'wgan_noise':log_wgan_noise,
			'mse_noise_sample':log_mse_noise_sample,
			'wgan_noise_sample':log_wgan_noise_sample,
			'mse_with_wgan_noise_sample':log_mse_with_wgan_noise_sample,
			'mae_with_wgan_noise_sample':log_mae_with_wgan_noise_sample,
		}


		f_full_train = {}
		nodes_full_train = {}
		for curve_key in will_draw_curve:
			if will_draw_curve[curve_key] == True:
				f_full_train[curve_key] = open(log_full_train[curve_key])
				nodes_full_train[curve_key] = []


		for curve_key in f_full_train:
			for line in f_full_train[curve_key]:
				try:
					node = eval(line)
					nodes_full_train[curve_key].append(node)
				except:
					print 'error'


		for i in range(len(keys)):
			plt.figure()

			# arrange layout
			y_full_train = {}
			for curve_key in nodes_full_train:
				epoch_limit = 400 - full_train_start
				if curve_key in ['mle_only']:
					epoch_limit += full_train_start
				if epoch_limit >= len(nodes_full_train[curve_key]):
					epoch_limit = -1
				y_full_train[curve_key] = np.array(nodes_full_train[curve_key][epoch_limit][keys[i]])#np.array([float(keys[i](node)) for node in nodes_full_train[curve_key]])


			# draw curve
			for curve_key in y_full_train:
				plt.plot(np.arange(1,len(y_full_train[curve_key])+1),y_full_train[curve_key],c=colors[curve_key],lw=1.2,
					label=labels_suffix[curve_key] + ' (on test.)')
				plt.scatter(np.arange(1,len(y_full_train[curve_key])+1),y_full_train[curve_key],c=colors[curve_key],lw=0)


			plt.xlabel('years')
			plt.title('metrics by ' + labels_prefix[i])
			plt.legend(loc=legend_loc[i])
			plt.gcf().set_size_inches(5.9, 5., forward=True)

			#plt.show()
			if i == 0: key = '' + dataset_id + '_gan_fulltrain_contrast_mape_acc_ACC.png'
			if i == 1: key = '' + dataset_id + '_gan_fulltrain_contrast_mape_acc_MAPE.png'
			# plt.xticks(fontsize=13)
			# plt.yticks(fontsize=13)
			# plt.legend(fontsize=13)
			plt.savefig(root + '/pic/%s'%key)

def print_full_train_contrast_acc_epsilon(dataset_id, nb_type=1):
	will_print = True
	will_print_curve = {
		'mle_only':True,
		'mse_noise':False,
		'wgan_noise':False,
		'mse_noise_sample':False,
		'wgan_noise_sample':True,
		'mse_with_wgan_noise_sample':True,
		'mae_with_wgan_noise_sample':True,
	}
	# preprocess
	dataset_path = root + '/data/' + dataset_id + '.txt'

	# pre-training
	ratio = [1,1]
	log_mle_only = root + '/data/' + dataset_id + '.pretrain.log.' + str(ratio[0]) + '.' + str(ratio[1]) + '.txt'

	# full-training
	# log_mle_to_mse = root + '/data/' + dataset_id + '.fulltrain.mse.log.txt'

	log_pre_train_early_stop = root + '/data/' + dataset_id + '.pretrain.early_stop.stop_point.json'
	with open(log_pre_train_early_stop) as fr:
		result = json.load(fr)
		ratio_key = str(ratio[0]) + ':' + str(ratio[1])
		early_stop = result[dataset_id][ratio_key]['mape_val']['stop_point']

	full_train_start = early_stop
	assert full_train_start > 0

	log_mse_noise = root + '/data/' + dataset_id + '.fulltrain.mse_noise.log.txt'
	log_wgan_noise = root + '/data/' + dataset_id + '.fulltrain.wgan_noise.log.txt'
	log_mse_noise_sample = root + '/data/' + dataset_id + '.fulltrain.mse_noise_sample.log.txt'
	log_wgan_noise_sample = root + '/data/' + dataset_id + '.fulltrain.wgan_noise_sample.log.txt'
	log_mse_with_wgan_noise_sample = root + '/data/' + dataset_id + '.fulltrain.mse_wgan.noise_sample.log.txt'
	log_mae_with_wgan_noise_sample = root + '/data/' + dataset_id + '.fulltrain.mae_wgan.noise_sample.log.txt'


	# drawing
	if will_print == True :
		colors = {
			'mle_only':'red',
			'mse_noise':'blue',
			'wgan_noise':'green',
			'mse_noise_sample':'purple',
			'wgan_noise_sample':'orange',
			'mse_with_wgan_noise_sample':'blue',
			'mae_with_wgan_noise_sample':'purple',
			'early_stop':'green',
		}
		keys = ['acc','mape']
		labels_prefix = ['ACC','MAPE']
		legend_loc = ['upper right','lower right']
		labels_suffix = {
			'mle_only':'MLE pretrain',
			'mse_noise':'ppGAN 1',
			'wgan_noise':'ppGAN 2',
			'mse_noise_sample':'ppGAN 3',
			'wgan_noise_sample':'MSE',
			'mse_with_wgan_noise_sample':'MSE+WGAN',
			'mae_with_wgan_noise_sample':'MAE+WGAN',
		}
		log_full_train = {
			'mle_only':log_mle_only,
			'mse_noise':log_mse_noise,
			'wgan_noise':log_wgan_noise,
			'mse_noise_sample':log_mse_noise_sample,
			'wgan_noise_sample':log_wgan_noise_sample,
			'mse_with_wgan_noise_sample':log_mse_with_wgan_noise_sample,
			'mae_with_wgan_noise_sample':log_mae_with_wgan_noise_sample,
		}


		f_full_train = {}
		nodes_full_train = {}
		for curve_key in will_print_curve:
			if will_print_curve[curve_key] == True:
				f_full_train[curve_key] = open(log_full_train[curve_key])
				nodes_full_train[curve_key] = []


		for curve_key in f_full_train:
			for line in f_full_train[curve_key]:
				try:
					node = eval(line)
					nodes_full_train[curve_key].append(node)
				except:
					print 'error'


		#for i in range(len(keys)):

		# arrange layout
		y_full_train = {}
		y_full_train_mape= {}
		for curve_key in nodes_full_train:
			epoch_limit = 400 - full_train_start
			if curve_key in ['mle_only']:
				epoch_limit += full_train_start
			if epoch_limit >= len(nodes_full_train[curve_key]):
				epoch_limit = -1
			y_full_train[curve_key] = nodes_full_train[curve_key][epoch_limit]['acc_vary']
			y_full_train_mape[curve_key] = nodes_full_train[curve_key][epoch_limit]['mape']


		# draw curve
		result = ''
		result += 'MAPE'
		min_mape = {}
		for curve_key in y_full_train_mape:
			for year in [5,10]:
				value = str(float('%.4f'%y_full_train_mape[curve_key][year-1]))
				if min_mape.has_key(str(year)) and min_mape[str(year)][1] > float(value):
					min_mape[str(year)] = [curve_key,float(value)]
				if not min_mape.has_key(str(year)) :
					min_mape[str(year)] = [curve_key,float(value)]

		for curve_key in y_full_train_mape:
			for year in [5,10]:
				value = str(float('%.4f'%y_full_train_mape[curve_key][year-1]))
				if curve_key == min_mape[str(year)][0]:
					result += '&\\textbf{' + value + '}'
				else:
					result += '&' + value
		result += '\\\\\n'

		for epsilon in [0.35,0.3,0.2,0.1]:
			max_acc = {}
			for curve_key in y_full_train:
				for year in [5,10]:
					value = str(float('%.4f'%y_full_train[curve_key][str(epsilon)][year-1]))
					if max_acc.has_key(str(year)) and max_acc[str(year)][1] < float(value):
						max_acc[str(year)] = [curve_key,float(value)]
					if not max_acc.has_key(str(year)) :
						max_acc[str(year)] = [curve_key,float(value)]

			result += 'ACC($\\epsilon$=' + str(epsilon) + ')'
			for curve_key in y_full_train:
				for year in [5,10]:
					value = str(y_full_train[curve_key][str(epsilon)][year-1])
					if curve_key == max_acc[str(year)][0]:
						result += '&\\textbf{' + value + '}'
					else:
						result += '&' + value
			result += '\\\\\n'
			# plt.plot(np.arange(1,len(y_full_train[curve_key])+1),y_full_train[curve_key],c=colors[curve_key],lw=1.2,
			#     label=labels_suffix[curve_key] + ' (on test.)')
			# plt.scatter(np.arange(1,len(y_full_train[curve_key])+1),y_full_train[curve_key],c=colors[curve_key],lw=0)


		key = '' + dataset_id + '.fulltrain.contrast.acc_epsilon.txt'
		with open(root + '/data/%s'%key,'w') as fw:
			fw.write(result)
		with open(root + '/data/%s'%key) as f:
			print f.read()



if __name__ == '__main__' :
	screen()
	event_types = {
		'paper3':1,
		'paper4':2, # paper4 is duplicate of paper3, while is interpreted in different way
		'patent3':1,
		'patent4':2, # patent4 is duplicate of patent3
		'atmerror2':1,
	}
	for dataset_id in ['atmerror2']:
		# draw_fix_train_non_self_m_hawkes(dataset_id,nb_type=event_types[dataset_id])
		draw_pretrain_learning_generator_convergence(dataset_id,nb_type=event_types[dataset_id])
		# draw_full_train_learning_gan_convergence(dataset_id,nb_type=event_types[dataset_id])
		# draw_full_train_learning_discriminative_convergence(dataset_id,nb_type=event_types[dataset_id])
		# draw_full_train_contrast_mape_acc(dataset_id,nb_type=event_types[dataset_id])
		# print_full_train_contrast_acc_epsilon(dataset_id,nb_type=event_types[dataset_id])
		pass
	plt.show()
