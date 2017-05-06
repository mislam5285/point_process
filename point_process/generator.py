#coding:utf-8
import numpy
import numpy as np
import os,sys,time
import json,csv
import cPickle as pickle
import copy
numpy.random.seed(1337)


class HawkesGenerator(object):
	def __init__(self):
		self.nb_type = None
		self.delta = 1.
		self.params = {}
		self.sequence_weights = None
		self.model = None
		self.hawkes_layer = None

	def pre_train(self,sequences,features,publish_years,pids,threshold,cut=15,predict_year=None,
			max_iter=0,max_outer_iter=100,alpha_iter=3,w_iter=30,val_length=5,early_stop=None,keep_weight_sync=True):
		""" 
			cut == observed length == T / delta, is a non-dimensional value
			predict_year is the firt year to begin predict
			cut_point == train_time == T == pred_year - pub_year == cut * delta, where delta is the value of scale (should be 1).
			At least one out of cut and predict_year should be specified.
		"""
		if cut is None:
			T = numpy.array([predict_year - publish_year for publish_year in publish_years],dtype=float)
			train_times = T
		else :
			T = numpy.array([float(cut)] * len(publish_years),dtype=float)
			train_times = T
			predict_year = -1

		[train_count,num_feature] = [len(features),len(features[0])] 
		nb_type = self.nb_type
		Z = numpy.asmatrix(numpy.ones([num_feature,nb_type])) #numpy.mat([[1.0] * nb_type]*num_feature)
		U = numpy.asmatrix(numpy.zeros([num_feature,nb_type])) #numpy.mat([[0.0] * nb_type]*num_feature)
		
		beta = numpy.asmatrix(numpy.ones([num_feature,nb_type])) #numpy.mat([[1.0] * nb_type]*num_feature)
		Alpha = numpy.ones([train_count,nb_type,nb_type]).tolist() #[[[1.0] * nb_type] * nb_type]*train_count
		# sw1 = 0.05
		W1 = (numpy.zeros([train_count,nb_type]) + 0.05).tolist() #[[0.05] * nb_type]*train_count
		W2 = numpy.ones([train_count,nb_type]).tolist() #[[1.0] * nb_type]*train_count
		# sw2 = 1.0

		init_time = time.time()
		init_clock = time.clock()

		likelihood = self.compute_likelihood(beta,train_count,features,W1,W2,Alpha,sequences,train_times)
		self.update_params(pids,Alpha,features,sequences,publish_years,predict_year,beta,W1,W2,cut=cut)
		mape_acc = self.compute_mape_acc(self.params,sequences,features,publish_years,pids,threshold,cut=cut)
		mape_acc_val = self.compute_mape_acc(self.params,sequences,features,publish_years,pids,threshold,
			duration=val_length,pred_year=predict_year-val_length,cut=cut-val_length)

		iterations = 1
		print {
			'source':'initial',
			'outer_iter':0,
			'inner_iter':0,
			'time':time.time() - init_time,
			'clock':time.clock() - init_clock,
			'LL':likelihood,
			'w1':numpy.mean(W1),
			'w2':numpy.mean(W2),
			'mean_alpha':numpy.mean(Alpha),
			'mean_beta':numpy.mean(beta),
			'mape':mape_acc['mape'],
			'acc':mape_acc['acc'],
			'mape_val':mape_acc_val['mape'],
			'acc_val':mape_acc_val['acc'],
		}
		sys.stdout.flush()

		for outer_times in range(max_outer_iter):

			# print 'step 1 : update Alpha,beta ...'

			for times in range(alpha_iter):
				D = numpy.mat([0.0]*num_feature) 
				E = numpy.mat([0.0]*num_feature)
				for sam in range(train_count): 
					s = sequences[sam]
					s = [x for x in s if x[0] < train_times[sam]]
					n = len(s)
					fea = numpy.mat(features[sam])
					sw1 = W1[sam]
					sw2 = W2[sam]
					salpha = Alpha[sam]

					for inner_iter in range(max_iter+1):
						# E-step
						p1 = numpy.multiply(beta,fea) / (beta * fea.T) 
						p2 = 0
						old_sum2 = 0
						for i in range(1,n): 
							mu = beta * fea.T * numpy.exp(- sw1 * s[i])
							sum1 = mu[0,0]
							sum2 = (old_sum2 + salpha) * numpy.exp(- sw2 * (s[i] - s[i-1]))
							old_sum2 = sum2
							summ = sum1 + sum2
							p1 += numpy.multiply(beta, fea) * numpy.exp(- sw1 * s[i]) / float(summ)
							p2 += 1 - mu/float(summ)
						# M-step
						alpha1 = p2
						alpha2 = (n - numpy.sum(numpy.exp(- sw2 * (T[sam] - numpy.array(s)))))/float(sw2)
						salpha = (alpha1/float(alpha2))[0,0]

					D += p1 
					E += fea * (1 - numpy.exp(- sw1 * T[sam])) / float(sw1) 
					Alpha[sam] = salpha

				penalty = 5.
				reg_beta = 20.
				# update beta, beta = ( - E + sqrt(E**2 + 4 * penalty * D) ) / (2 * penalty)
				for find in range(num_feature): 
					_E = E[0,find] + penalty * (U[0,find] - Z[0,find]) 
					beta[0,find] = (numpy.sqrt(_E**2 + 4*penalty*D[0,find]) - _E) /float(2*penalty)
				
				# z-update without relaxation
				Z = self.shrinkage(beta+U, reg_beta/float(penalty)) 
				U = U + beta - Z 

				likelihood = self.compute_likelihood(beta,train_count,features,W1,W2,Alpha,sequences,train_times)
				self.update_params(pids,Alpha,features,sequences,publish_years,predict_year,beta,W1,W2,cut=cut)
				mape_acc = self.compute_mape_acc(self.params,sequences,features,publish_years,pids,threshold,cut=cut)
				mape_acc_val = self.compute_mape_acc(self.params,sequences,features,publish_years,pids,threshold,
					duration=val_length,pred_year=predict_year-val_length,cut=cut-val_length)

				print {
					'source':'update Alpha beta',
					'outer_iter':outer_times,
					'inner_iter':times,
					'time':time.time() - init_time,
					'clock':time.clock() - init_clock,
					'LL':likelihood,
					'w1':numpy.mean(W1),
					'w2':numpy.mean(W2),
					'mean_alpha':numpy.mean(Alpha),
					'mean_beta':numpy.mean(beta),
					'mape':mape_acc['mape'],
					'acc':mape_acc['acc'],
					'mape_val':mape_acc_val['mape'],
					'acc_val':mape_acc_val['acc'],
				}
				sys.stdout.flush()
				iterations += 1
				if early_stop is not None and iterations >= early_stop: break

			# print 'step 2 : update w by gradient descent ...'


			step_size = 1e-2
			for times in range(w_iter):
				step_size /= 1 + 10 * step_size
				for sam in range(train_count):
					s = sequences[sam]
					s = numpy.mat([x for x in s if x[0] < train_times[sam]])
					n = s.shape[1]
					fea = numpy.mat(features[sam])
					sw1 = W1[sam]
					sw2 = W2[sam]
					salpha = Alpha[sam]
					old_obj = 0
					count = 0
					# while 1:
					pw1 = -s[0,0]
					pw2 = numpy.mat(0.0)
					old_sum2 = 0
					for i in range(1,n):
						mu = beta * fea.T * numpy.exp(- sw1 * s[0,i])
						sum1 = mu
						sum2 = (old_sum2 + salpha) * numpy.exp(- sw2 *(s[0,i] - s[0,i-1]))
						old_sum2 = sum2
						summ = sum1 + sum2;
						pw2t = salpha * numpy.sum(numpy.multiply(numpy.exp(- sw2 * \
							(s[0,i] - s[0,0:i])),-(s[0,i]-s[0,0:i])))
						pw1 = pw1 + beta * fea.T * numpy.exp(- sw1 * s[0,i]) * (-s[0,i]) / float(summ)
						pw2 = pw2 + pw2t / float(summ)
					pw1 = pw1 - beta * fea.T * (numpy.exp(-sw1*T[sam]) * T[sam] * sw1 - \
						(1 - numpy.exp(-sw1*T[sam]))) /float(sw1**2)
					upper = numpy.multiply(numpy.exp(-sw2*(T[sam]-s)),((T[sam]-s)*sw2))
					lower = (1-numpy.exp(-sw2*(T[sam]-s)))

					pw2 = pw2 - salpha*numpy.sum(( upper - lower )/float(sw2**2))
					sw1 += step_size*numpy.sign(pw1)[0,0]
					sw2 += step_size*numpy.sign(pw2)[0,0]
					W1[sam] = sw1
					W2[sam] = sw2

				likelihood = self.compute_likelihood(beta,train_count,features,W1,W2,Alpha,sequences,train_times)
				self.update_params(pids,Alpha,features,sequences,publish_years,predict_year,beta,W1,W2,cut=cut)
				mape_acc = self.compute_mape_acc(self.params,sequences,features,publish_years,pids,threshold,cut=cut)
				mape_acc_val = self.compute_mape_acc(self.params,sequences,features,publish_years,pids,threshold,
					duration=val_length,pred_year=predict_year-val_length,cut=cut-val_length)

				print {
					'source':'update w1 w2',
					'outer_iter':outer_times,
					'inner_iter':times,
					'time':time.time() - init_time,
					'clock':time.clock() - init_clock,
					'LL':likelihood,
					'w1':numpy.mean(W1),
					'w2':numpy.mean(W2),
					'mean_alpha':numpy.mean(Alpha),
					'mean_beta':numpy.mean(beta),
					'mape':mape_acc['mape'],
					'acc':mape_acc['acc'],
					'mape_val':mape_acc_val['mape'],
					'acc_val':mape_acc_val['acc'],
				}
				sys.stdout.flush()
				iterations += 1
				if early_stop is not None and iterations >= early_stop: break

			# print 'step 3 : check terminate condition ...'

			if early_stop is not None and iterations >= early_stop: break

		self.update_sequence_weights(pids,Alpha,features,sequences,publish_years,predict_year,beta,W1,W2)
		return self.params

	def update_params(self,pids,Alpha,features,sequences,publish_years,predict_year,beta,W1,W2,cut=None):

		patent = {}
		for item in range(len(Alpha)):
			a_patent = {}
			a_patent['alpha'] = Alpha[item]
			a_patent['w1'] = W1[item]
			a_patent['w2'] = W2[item]
			a_patent['fea'] = features[item]
			a_patent['cite'] = sequences[item]
			a_patent['year'] = publish_years[item]
			patent[pids[item]] = a_patent

		params = {}
		params['predict_year'] = predict_year
		params['cut_point'] = cut
		params['train_count'] = len(patent)
		params['beta'] = beta.tolist()
		params['patent'] = patent
		params['feature_name'] = []

		self.params = params

	def update_sequence_weights(self,pids,Alpha,features,sequences,publish_years,predict_year,beta,W1,W2):

		result = []
		for i in range(len(pids)):
			seq = {}
			fea = numpy.mat(features[i])
			beta = numpy.mat(beta)
			seq['seq_id'] = i
			seq['paper_id'] = pids[i]
			seq['theta'] = W1[i]
			seq['w'] = W2[i]
			seq['alpha'] = Alpha[i]
			seq['fea'] = features[item]
			seq['beta'] = beta.tolist()
			seq['spont'] = (fea*beta).tolist()[0]
			result.append(seq)

		self.sequence_weights = result


	def compute_likelihood(self,beta,train_count,features,W1,W2,Alpha,sequences,train_times):
		likelihood = 0.
		for item in range(train_count):
			fea = numpy.mat(features[item])
			sw1 = W1[item]
			sw2 = W2[item]
			salpha = Alpha[item]
			s = sequences[item]
			s = [x for x in s if x[0] < train_times[item]]
			obj = self.calculate_objective((fea*beta).tolist()[0],sw1,salpha,sw2,s,train_times[item])
			likelihood -= obj
		likelihood /= train_count
		return likelihood


	def shrinkage(self,vector,kappa):
		mat1 = vector-kappa
		mat2 = -vector-kappa
		for i in range(mat1.shape[0]):
			for j in range(mat1.shape[1]):
				if mat1[i,j] < 0 :
					mat1[i,j] = 0
				if mat2[i,j] < 0 :
					mat2[i,j] = 0
		return mat1 - mat2

	def calculate_objective(self,spont,sw1,salpha,sw2,events,train_time):
		T = train_time
		N = len(events)
		s = events
		spontaneous = spont[s[0][1]]
		w1 = sw1[s[0][1]]
		obj = numpy.log(spontaneous*numpy.exp(-w1*s[0][0]))

		old_sum2 = 0.
		for i in range(1,N):
			spontaneous = spont[s[i][1]]
			w1 = sw1[s[i][1]]
			w2 = sw2[s[i][1]]
			alpha = salpha[s[i][1]][s[i-1][1]]
			mu = spontaneous*numpy.exp(-w1*s[i][0])
			sum1 = mu
			sum2 = (old_sum2 + alpha)*numpy.exp(-w2*(s[i][0]-s[i-1][0]))
			old_sum2 = sum2
			obj += numpy.log(sum1+sum2)

		for m in range(self.nb_type):
			w2 = sw2[m]
			activate_sum = 0.
			for i in range(N):
				alpha = salpha[m][s[i][1]]
				activate = numpy.exp(-w2*(T-s[i][0]))
				activate_sum += (1. - activate) * alpha / float(w2)
			obj -= activate_sum
		# activate = numpy.exp(-w2*(T-(numpy.mat(s)[:,0])))
		# activate_sum = numpy.sum((1-activate))*alpha/float(w2)
		# obj= obj - activate_sum

		for m in range(self.nb_type):
			spontaneous = spont[m]
			w1 = sw1[m]
			obj -= (spontaneous/w1) * (1 - numpy.exp(-w1*T))

		return obj

	# def calculate_objective_single(self,spontaneous,w1,alpha,w2,events,train_time):
	# 	T=train_time
	# 	N=len(events)
	# 	s=events
	# 	old_sum2 = 0
	# 	obj = numpy.log(spontaneous*numpy.exp(-w1*s[0]))
	# 	for i in range(1,N):
	# 		mu = spontaneous*numpy.exp(-w1*s[i])
	# 		sum1 = mu
	# 		sum2 = (old_sum2 + alpha)*numpy.exp(-w2*(s[i]-s[i-1]))
	# 		old_sum2 = sum2
	# 		obj=obj+numpy.log(sum1+sum2)
	# 	activate = numpy.exp(-w2*(T-numpy.mat(s)))
	# 	activate_sum = numpy.sum((1-activate))*alpha/float(w2)
	# 	obj= obj - activate_sum 
	# 	obj = obj - (spontaneous/w1) * (1 - numpy.exp(-w1*T))
	# 	return obj

	def compute_mape_acc(self,model,sequences,features,publish_years,pids,threshold,duration=10,pred_year=None,cut=None):
		patents = self.params['patent']
		g_z = []
		x = []
		for i, key in enumerate(patents):
			if cut is None:
				pred_year = pred_year or self.params['predict_year']
			else:
				pred_year = cut + int(float((patents[key]['year'])))
			real = self.real_one(key,cut_point=pred_year - int(float((patents[key]['year']))) + duration)
			pred = self.predict_one(key,duration=duration,pred_year=pred_year)
			x.append(real)
			g_z.append(pred)

		x = numpy.array(x)
		g_z = numpy.array(g_z)
		assert g_z.shape == x.shape
		count_g_z = np.sum(g_z,2)
		count_x = np.sum(x,2)
		for i in range(1,g_z.shape[1]):
			count_g_z[:,i] += count_g_z[:,i-1]
			count_x[:,i] += count_x[:,i-1]
		count_g_z = count_g_z[:,-duration:]
		count_x = count_x[:,-duration:]

		mape = np.mean(np.abs(count_g_z - count_x)/(count_x + 0.1),0)
		acc = np.mean(np.abs(count_g_z - count_x)/(count_x + 0.1) < 0.3,0)

		return {
			'mape':mape.tolist(),
			'acc':acc.tolist(),
		}

	# def compute_mape_acc_single(self,model,sequences,features,publish_years,pids,threshold,duration=10,pred_year=None,cut=None):
	# 	patents = self.params['patent']
	# 	diffs = []
	# 	ll = 20
	# 	for i, key in enumerate(patents):
	# 		if cut is None:
	# 			pred_year = pred_year or self.params['predict_year']
	# 		else:
	# 			pred_year = cut + int(float((patents[key]['year'])))
	# 		real = self.real_one(key)
	# 		pred = self.predict_one(key,duration=duration,pred_year=pred_year)
	# 		diff = []
	# 		ir = 0 # index_real
	# 		for p in pred:
	# 			while int(real[ir][0]) < int(p[0]):
	# 				ir += 1
	# 				if ir >= len(real):
	# 					ir = len(real) - 1
	# 					break
	# 			# if ir != len(real) - 1 and int(real[ir][0]) != int(p[0]):
	# 			# 	ir -= 1
	# 			diff.append((p[0],(p[1] - real[ir][1])/float(real[ir][1] + 0.1) ))
	# 		diffs.append(diff)
	# 		if ll > len(diff) : ll = len(diff)
	# 	mape = [0.0] * ll
	# 	acc = [0.0] * ll
	# 	for diff in diffs:
	# 		for i in range(ll):
	# 			mape[i] += abs(diff[i][1])
	# 			if abs(diff[i][1]) < 0.3 :
	# 				acc[i] += 1
	# 	for i in range(ll):
	# 		mape[i] /= float(len(diffs))
	# 		acc[i] /= float(len(diffs))
		
	# 	return {'mape':mape,'acc':acc}

	def predict_one(self,_id,duration,pred_year):
		try:
			patent = self.params['patent'][str(_id)]
		except KeyError,e:
			return None
		sw1 = patent['w1']
		salpha = patent['alpha']
		sw2 = patent['w2']
		fea = numpy.mat(patent['fea'])
		ti = patent['cite']
		beta = numpy.mat(self.params['beta'])

		cut_point = pred_year - int(float((patent['year'])))
		tr = [x for x in ti if x[0] < cut_point]

		return self.predict_year_by_year(tr,cut_point,duration,
			(fea*beta).tolist()[0],sw1,salpha,sw2)

	def real_one(self,_id,cut_point):
		try:
			patent = self.params['patent'][str(_id)]
		except KeyError,e:
			return None

		cites = patent['cite']
		N = len(cites)
		real_seq = numpy.zeros([cut_point,self.nb_type]).tolist()

		# copy unit
		left = 0
		for t in range(cut_point):
			while left < N and cites[left][0] < t + 1:
				real_seq[t][cites[left][1]] += 1.
				left += 1

		return real_seq


	# def predict_one_single(self,_id,duration,pred_year):
	# 	try:
	# 		patent = self.params['patent'][str(_id)]
	# 	except KeyError,e:
	# 		return None
	# 	sw1 = patent['w1']
	# 	salpha = patent['alpha']
	# 	sw2 = patent['w2']
	# 	fea = numpy.mat(patent['fea'])
	# 	ti = patent['cite']
	# 	beta = numpy.mat(self.params['beta'])

	# 	cut_point = pred_year - int(float((patent['year'])))
	# 	tr = [x for x in ti if x[0] < cut_point]

	# 	pred = self.predict_year_by_year(tr,cut_point,duration,
	# 		(fea*beta).tolist()[0],sw1,salpha,sw2)

	# 	_dict = {}
	# 	for i in range(len(pred)):
	# 		year = pred_year + i
	# 		_dict[year] = pred[i]
	# 	_list = sorted(_dict.items(),key=lambda x:x[0])
	# 	return _list

	# def real_one_single(self,_id):
	# 	try:
	# 		patent = self.params['patent'][str(_id)]
	# 	except KeyError,e:
	# 		return None

	# 	cites = patent['cite']
	# 	_dict = {}
	# 	counts = [0.] * self.nb_type
	# 	for i in range(len(cites)):
	# 		counts[cites[i][1]] += 1.
	# 		year = int(float(cites[i][0])) + int(float(patent['year']))
	# 		_dict[year] = copy.copy(counts)
	# 	_list = sorted(_dict.items(),key=lambda x:x[0])
	# 	return _list

	def predict_year_by_year(self,tr,cut_point,duration,spont,sw1,salpha,sw2):
		N = len(tr)
		pred_seq = numpy.zeros([cut_point,self.nb_type]).tolist()

		# copy unit
		left = 0
		for t in range(cut_point):
			while left < N and tr[left][0] < t + 1:
				pred_seq[t][tr[left][1]] += 1.
				left += 1

		# prediction_unit
		spont = numpy.mat(spont)
		theta = numpy.mat(sw1)
		w = numpy.mat(sw2)
		alpha = numpy.mat(salpha)
		for t in range(cut_point,cut_point+duration):
			term1 = numpy.multiply(spont/theta,
				(numpy.exp(-theta*t) - numpy.exp(-theta*(t + 1.))))
			# triggering_unit
			effect = numpy.mat([0.] * self.nb_type)
			for tao in range(t):
				counts = numpy.mat(pred_seq[tao])
				effect_unit = numpy.multiply(counts,
					(numpy.exp(- w * (t - tao)) - numpy.exp(- w * (t + 1 - tao))))
				effect += effect_unit
			term2 = (alpha * (effect.T)).T / w
			pred_seq.append((term1 + term2).tolist()[0])

		return pred_seq


	# def predict_year_by_year_single(self,tr,cut_point,duration,spont,sw1,salpha,sw2):
	# 	N = tr.shape[1] 
	# 	pred = []
	# 	for t in range(cut_point+1,cut_point+duration+1):
	# 		delta_ct = spontaneous/w1*(numpy.exp(-w1*(t-1))-numpy.exp(-w1*t)) + \
	# 			alpha/w2*(numpy.sum(numpy.exp(-w2*((t-1)-tr)))-numpy.sum(numpy.exp(-w2*(t-tr))))
	# 		delta_ct = delta_ct[0,0]
	# 		if len(pred) == 0:
	# 			ct = N + delta_ct
	# 		else :
	# 			ct = pred[-1] + delta_ct
	# 		tr = tr.tolist()[0]
	# 		tr.extend([t for i in range(int(delta_ct))])
	# 		tr = numpy.mat(tr)
	# 		pred.append(ct)
	# 	return pred

	def create_trainable_model(self,sequences, pred_length, proxy_layer=None, need_noise_dropout=False, stddev=5.,sample_stddev=None):
		from keras.layers import Input, GaussianNoise
		from keras.models import Model

		from pp_layer import HawkesLayer

		x = Input(batch_shape=(1,1), dtype='int32')
		hawkes_layer = HawkesLayer(sequences,pred_length,sequence_weights=self.sequence_weights,proxy_layer=proxy_layer,sample_stddev=sample_stddev)
		y = hawkes_layer(x)
		if need_noise_dropout == True:
			y = GaussianNoise(stddev)(y)

		model = Model(inputs=[x], outputs=[y], name='hawkes_output')

		self.model = model
		self.hawkes_layer = hawkes_layer
		return model

	def load(self,f,nb_type=1):
		"""
			data format : types | start time | feature
			the value of time scale should be 1. 
			if the distance between adjacent scale mark is too long, it should be splitted.
		"""
		self.nb_type = nb_type
		if nb_type > 1:
			data = []
			pids = []
			span = nb_type + 2
			for i,row in enumerate(csv.reader(file(f,'r'))):
				if i % span == span - 2: # start time
					pids.append(str(row[0]))
					row = [float(row[1])]
				elif i % span < span - 2: # event types
					row = [float(x) for x in row[1:]]
				elif i % span == span - 1: # profile feature
					_row = [float(x) for x in row[1:]]
					_max = max(_row)
					_min = min(_row)
					row = [(x - _min)/float(_max - _min) for x in _row]
				data.append(row)
			
			I = int(len(data)/span)
			train_seq = []
			test_seq = []
			sequences = []
			features = []
			publish_years = []
			for i in range(I):
				publish_year = data[i * span + span - 2][0]
				feature = data[i * span + span - 1]
				# time_seq = self_seq + nonself_seq
				# time_seq.sort()
				sequence = data[(i*span):(i*span + span - 2)]

				for j in range(span - 2):
					sequence[j] = [float(int(x)) for x in sequence[j] if x > -1.]

				time_seq = []
				for k,seq in enumerate(sequence):
					time_seq += [(x,k) for x in seq]
				time_seq.sort(key=lambda x:x[0])

				sequences.append(time_seq)
				features.append(feature)
				publish_years.append(publish_year)

			threshold = 0.01
			return sequences,features,publish_years,pids,threshold
		else:
			span = 0
			key_prev = ''
			for i,row in enumerate(csv.reader(file(f,'r'))):
				key = str(row[0])
				if key == key_prev or key_prev == '': 
					span += 1
					key_prev = key
				else:
					break

			data = []
			pids = []
			for i,row in enumerate(csv.reader(file(f,'r'))):
				if i % span == span - 2: # start time
					pids.append(str(row[0]))
					row = [float(row[1])]
				elif i % span < span - 2: # event types
					row = [float(x) for x in row[1:]]
				elif i % span == span - 1: # profile feature
					_row = [float(x) for x in row[1:]]
					_max = max(_row)
					_min = min(_row)
					row = [(x - _min)/float(_max - _min) for x in _row]
				data.append(row)
			
			I = int(len(data)/span)
			train_seq = []
			test_seq = []
			sequences = []
			features = []
			publish_years = []
			for i in range(I):
				publish_year = data[i * span + span - 2][0]
				feature = data[i * span + span - 1]
				# time_seq = self_seq + nonself_seq
				# time_seq.sort()
				sequence = data[(i*span):(i*span + span - 2)]

				for j in range(span - 2):
					sequence[j] = [float(int(x)) for x in sequence[j] if x > -1.]

				time_seq = []
				for seq in sequence:
					time_seq += [(x,0) for x in seq]
				time_seq.sort(key=lambda x:x[0])

				sequences.append(time_seq)
				features.append(feature)
				publish_years.append(publish_year)

			threshold = 0.01
			return sequences,features,publish_years,pids,threshold


class RNNGenerator(object):
	def __init__(self):
		self.params = {}
		self.l1 = 1.
		self.l2 = 1.

	def train(self,sequences,labels,features,publish_years,pids,superparams,cut=None,predict_year=2000,max_iter=0,max_outer_iter=100):
		from keras.layers import Input, Dense, Masking, LSTM, Activation, Dropout, merge
		from keras.models import Model
		from keras.regularizers import l1,l2

		f = Input(shape=(1,len(features[0])),dtype='float')
		# features[paper][feature], sequences[paper][day][feature]
		k = Dense(len(sequences[0][0]),activation='relu',W_regularizer=l1(self.l1))(f)
		# k = merge([k,k],mode='concat',concat_axis=1)
		k1 = Dropout(0.5)(k)
		k2 = Input(shape=(len(sequences[0]),len(sequences[0][0])),dtype='float')
		g1 = merge([k1,k2],mode='concat',concat_axis=1)
		m1 = Masking(mask_value= -1.)(g1)
		n1 = LSTM(len(sequences[0][0]),activation='relu',W_regularizer=l2(self.l2),dropout_W=0.5,dropout_U=0.5)(m1)
		# n1 = Dense(16)(n1)
		# n1 = Activation('tanh')(n1)
		# n1 = Dropout(0.5)(n1)
		# n1 = Dense(4)(n1)
		# n1 = Activation('tanh')(n1)
		# n1 = Dense(len(sequences[0][0]))(n1)

		model = Model(inputs=[f,k2], outputs=[n1])
		model.compile(optimizer='adam', loss='mape')
		model.fit(
			[numpy.array([[f] for f in features]),numpy.array(sequences)], [numpy.array(labels)],
			epochs=500, batch_size=10,
			verbose=1,validation_split=0.2)
		self.params['model'] = model
		# embeddings = model.layers[1].W.get_value()
		return model

	def load(self,f,cut=15):
		# features[paper][feature], sequences[paper][day][feature]
		data = []
		pids = []
		for i,row in enumerate(csv.reader(file(f,'r'))):
			if i % 4 == 2:
				pids.append(str(row[0]))
				row = [float(row[1])]
			elif i % 4 == 0 or i % 4 == 1:
				row = [float(x) for x in row[1:]]
			elif i % 4 == 3:
				_row = [float(x) for x in row[1:]]
				_max = max(_row)
				_min = min(_row)
				row = [(x - _min)/float(_max - _min) for x in _row]
			data.append(row)
		
		I = int(len(data)/4)
		#train_seq = []
		# test_seq = []
		sequences = []
		labels = []
		features = []
		publish_years = []
		for i in range(I):
			publish_year = data[i * 4 + 2][0]
			self_seq = data[i * 4]
			nonself_seq = data[i * 4 + 1]
			feature = data[i * 4 + 3]
			#time_seq = self_seq + nonself_seq
			#time_seq.sort()

			# sequences.append(time_seq)
			sequence = []
			for year in range(cut) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1])
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
				sequence.append([self_count,nonself_count])
			sequences.append(sequence)

			self_count = len([x for x in self_seq if cut <= x and x < cut + 1])
			nonself_count = len([x for x in nonself_seq if cut <= x and x < cut + 1])
			labels.append([self_count,nonself_count])

			features.append(feature)
			publish_years.append(publish_year)

		superparams = {}
		return sequences,labels,features,publish_years,pids,superparams	

	def load_events(self,f,cut=15):
		# features[paper][feature], sequences[paper][day][feature]
		data = []
		pids = []
		for i,row in enumerate(csv.reader(file(f,'r'))):
			if i % 4 == 2:
				pids.append(str(row[0]))
				row = [float(row[1])]
			elif i % 4 == 0 or i % 4 == 1:
				row = [float(x) for x in row[1:]]
			elif i % 4 == 3:
				_row = [float(x) for x in row[1:]]
				_max = max(_row)
				_min = min(_row)
				row = [(x - _min)/float(_max - _min) for x in _row]
			data.append(row)
		
		I = int(len(data)/4)
		#train_seq = []
		# test_seq = []
		sequences = []
		labels = []
		features = []
		publish_years = []
		for i in range(I):
			publish_year = data[i * 4 + 2][0]
			self_seq = data[i * 4]
			nonself_seq = data[i * 4 + 1]
			feature = data[i * 4 + 3]
			# time_seq = self_seq + nonself_seq
			# time_seq.sort()

			# sequences.append(time_seq)
			sequence = []
			interval = 1
			for year in range(cut) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1])
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
				if self_count + nonself_count == 0 :
					sequence.append([-1.,-1.,-1.])
					interval += 1
				else :
					sequence.append([self_count,nonself_count,interval])
					interval = 1
			sequences.append(sequence)


			self_count = len([x for x in self_seq if cut <= x and x < cut + 1])
			nonself_count = len([x for x in nonself_seq if cut <= x and x < cut + 1])
			cut2 = cut
			while self_count + nonself_count == 0 and cut2 < cut + 10:
				self_count = len([x for x in self_seq if cut2 <= x and x < cut2 + 1])
				nonself_count = len([x for x in nonself_seq if cut2 <= x and x < cut2 + 1])
				cut2 += 1
				interval += 1

			labels.append([self_count,nonself_count,interval])

			features.append(feature)
			publish_years.append(publish_year)

		superparams = {}
		return sequences,labels,features,publish_years,pids,superparams



if __name__ == '__main__':
	with open('../log/train.log','w') as f:
		sys.stdout = f
		if len(sys.argv) == 2:
			predictor = RNNGenerator()
			loaded = predictor.load('../data/paper3.txt')
			predictor.train(*loaded)
			exit()
		predictor = HawkesGenerator()
		# loaded = predictor.load('../data/paper3.txt')
		loaded = predictor.load('../data/paper3.txt',nb_type=2)
		# print loaded[0][0]
		# print loaded[0][1]
		# print loaded[0][2]
		model = predictor.pre_train(*loaded)
		

		pass		