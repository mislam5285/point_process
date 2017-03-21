#coding:utf-8
import numpy
import os,sys,time
import json,csv
import cPickle as pickle
numpy.random.seed(1337)


class HawkesGenerator(object):
	def __init__(self):
		self.params = {}

	def pre_train(self,sequences,features,publish_years,pids,threshold,cut=None,predict_year=2000,max_iter=0,max_outer_iter=100,alpha_iter=3,w_iter=30):

		if cut is None:
			T = numpy.array([predict_year - publish_year + 1 for publish_year in publish_years],dtype=float)
			train_times = T
		else :
			T = numpy.array([cut+0.001] * len(publish_years))
			train_times = T

		[train_count,num_feature] = [len(features),len(features[0])] 
		rho = 1
		lam = 2
		Z = numpy.mat([1.0]*num_feature)
		U = numpy.mat([0.0]*num_feature)
		
		beta = numpy.mat([1.0]*num_feature)  
		alpha = [1.0]*train_count
		# sw1 = 0.05
		W1 = [0.05]*train_count
		W2 = [1.0]*train_count
		# sw2 = 1.0

		init_time = time.time()
		init_clock = time.clock()

		likelihood = self.compute_likelihood(beta,train_count,features,W1,W2,alpha,sequences,train_times)
		self.update_params(pids,alpha,features,sequences,publish_years,predict_year,beta,W1,W2)
		mape_acc = self.predict(self.params,sequences,features,publish_years,pids,threshold)

		print {
			'source':'initial',
			'outer_iter':0,
			'inner_iter':0,
			'time':time.time() - init_time,
			'clock':time.clock() - init_clock,
			'LL':likelihood,
			'w1':numpy.mean(W1),
			'w2':numpy.mean(W2),
			'mean_alpha':numpy.mean(alpha),
			'mean_beta':numpy.mean(beta),
			'mape':mape_acc['mape'],
			'acc':mape_acc['acc'],
		}
		sys.stdout.flush()

		for outer_times in range(max_outer_iter):

			# print 'step 1 : update alpha,beta ...'

			for times in range(alpha_iter):
				v1 = numpy.mat([0.0]*num_feature) 
				v2 = numpy.mat([0.0]*num_feature)
				for sam in range(train_count): 
					s = sequences[sam]
					s = [x for x in s if x <= train_times[sam]]
					n = len(s)
					fea = numpy.mat(features[sam])
					sw1 = W1[sam]
					sw2 = W2[sam]
					salpha = alpha[sam]
					old_obj = 0 

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
						salpha = alpha1/float(alpha2)

					v1 += p1 
					v2 += fea * (1 - numpy.exp(- sw1 * T[sam])) / float(sw1) 
					alpha[sam] = salpha

				# update beta, beta = v1./v2;
				for find in range(num_feature): 
					B = v2[0,find] + rho * (U[0,find] - Z[0,find]) 
					beta[0,find] = (numpy.sqrt(B**2 + 4*rho*v1[0,find]) - B) /float(2*rho)
				
				# z-update without relaxation
				Z = self.shrinkage(beta+U, lam/float(rho)) 
				U = U + beta - Z 

				likelihood = self.compute_likelihood(beta,train_count,features,W1,W2,alpha,sequences,train_times)
				self.update_params(pids,alpha,features,sequences,publish_years,predict_year,beta,W1,W2)
				mape_acc = self.predict(self.params,sequences,features,publish_years,pids,threshold)

				print {
					'source':'update alpha beta',
					'outer_iter':outer_times,
					'inner_iter':times,
					'time':time.time() - init_time,
					'clock':time.clock() - init_clock,
					'LL':likelihood,
					'w1':numpy.mean(W1),
					'w2':numpy.mean(W2),
					'mean_alpha':numpy.mean(alpha),
					'mean_beta':numpy.mean(beta),
					'mape':mape_acc['mape'],
					'acc':mape_acc['acc'],
				}
				sys.stdout.flush()

			# print 'step 2 : update w by gradient descent ...'


			step_size = 1e-2
			for times in range(w_iter):
				step_size /= 1 + 10 * step_size
				for sam in range(train_count):
					s = sequences[sam]
					s = numpy.mat([x for x in s if x <= train_times[sam]])
					n = s.shape[1]
					fea = numpy.mat(features[sam])
					sw1 = W1[sam]
					sw2 = W2[sam]
					salpha = alpha[sam]
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

				likelihood = self.compute_likelihood(beta,train_count,features,W1,W2,alpha,sequences,train_times)
				self.update_params(pids,alpha,features,sequences,publish_years,predict_year,beta,W1,W2)
				mape_acc = self.predict(self.params,sequences,features,publish_years,pids,threshold)

				print {
					'source':'update w1 w2',
					'outer_iter':outer_times,
					'inner_iter':times,
					'time':time.time() - init_time,
					'clock':time.clock() - init_clock,
					'LL':likelihood,
					'w1':numpy.mean(W1),
					'w2':numpy.mean(W2),
					'mean_alpha':numpy.mean(alpha),
					'mean_beta':numpy.mean(beta),
					'mape':mape_acc['mape'],
					'acc':mape_acc['acc'],
				}
				sys.stdout.flush()

			# print 'step 3 : check terminate condition ...'

			if outer_times > max_outer_iter:
				break
			outer_times += 1

		self.update_params(pids,alpha,features,sequences,publish_years,predict_year,beta,W1,W2)
		return self.params

	def update_params(self,pids,alpha,features,sequences,publish_years,predict_year,beta,W1,W2):

		patent = {}
		for item in range(len(alpha)):
			a_patent = {}
			a_patent['alpha'] = alpha[item]
			a_patent['w1'] = W1[item]
			a_patent['w2'] = W2[item]
			a_patent['fea'] = features[item]
			a_patent['cite'] = sequences[item]
			a_patent['year'] = publish_years[item]
			patent[pids[item]] = a_patent

		params = {}
		params['predict_year'] = predict_year
		params['train_count'] = len(patent)
		params['beta'] = beta.tolist()[0]
		params['patent'] = patent
		params['feature_name'] = ["assignee_type",
								"n_inventor",
								"n_claim",
								"n_backward",
								"ratio_cite",
								"generality",
								"originality",
								"forward_lag",
								"backward_lag"]

		self.params = params


	def compute_likelihood(self,beta,train_count,features,W1,W2,alpha,sequences,train_times):
		likelihood = 0;
		for item in range(train_count):
			fea = numpy.mat(features[item])
			sw1 = W1[item]
			sw2 = W2[item]
			salpha = alpha[item]
			s = sequences[item]
			s = [x for x in s if x <= train_times[item]]
			obj = self.calculate_objective(beta*fea.T,sw1,salpha,sw2,s,train_times[item])
			likelihood -= likelihood - obj[0,0]
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

	def calculate_objective(self,spontaneous,w1,alpha,w2,events,train_times):
		T=train_times
		N=len(events)
		s=events
		old_sum2 = 0
		obj = numpy.log(spontaneous*numpy.exp(-w1*s[0]))
		for i in range(1,N):
			mu = spontaneous*numpy.exp(-w1*s[i])
			sum1 = mu
			sum2 = (old_sum2 + alpha)*numpy.exp(-w2*(s[i]-s[i-1]))
			old_sum2 = sum2
			obj=obj+numpy.log(sum1+sum2)
		activate = numpy.exp(-w2*(T-numpy.mat(s)))
		activate_sum = numpy.sum((1-activate))*alpha/float(w2)
		obj= obj - activate_sum 
		obj = obj - (spontaneous/w1) * (1 - numpy.exp(-w1*T))
		return obj

	def predict(self,model,sequences,features,publish_years,pids,threshold):
		patents = self.params['patent']
		diffs = []
		ll = 20
		for i, key in enumerate(patents):
			real = self.real_one(key)
			pred = self.predict_one(key,10,self.params['predict_year'])
			diff = []
			ir = 0 # index_real
			for p in pred:
				while int(real[ir][0]) < int(p[0]):
					ir += 1
					if ir >= len(real):
						ir = len(real) - 1
						break
				if ir != len(real) - 1 and int(real[ir][0]) != int(p[0]):
					ir -= 1
				diff.append((p[0],(p[1] - real[ir][1])/float(real[ir][1]) ))
			diffs.append(diff)
			if ll > len(diff) : ll = len(diff)
		mape = [0.0] * ll
		acc = [0.0] * ll
		for diff in diffs:
			for i in range(ll):
				mape[i] += abs(diff[i][1])
				if abs(diff[i][1]) < 0.3 :
					acc[i] += 1
		for i in range(ll):
			mape[i] /= float(len(diffs))
			acc[i] /= float(len(diffs))
		
		return {'mape':mape,'acc':acc}

	def predict_one(self,_id,duration,pred_year):
		try:
			patent = self.params['patent'][str(_id)]
		except KeyError,e:
			return None
		w1 = patent['w1']
		alpha = patent['alpha']
		w2 = patent['w2']
		fea = numpy.mat(patent['fea'])
		ti = patent['cite']
		beta = numpy.mat(self.params['beta'])

		cut_point = pred_year - int(float((patent['year'])))
		tr = numpy.mat([x for x in ti if x <= cut_point])
		pred = self.predict_year_by_year(tr,cut_point,duration,
			beta*numpy.mat(fea).T,w1,alpha,w2)

		_dict = {}
		for i in range(len(pred)):
			year = pred_year + i + 1
			_dict[year] = pred[i]
		_list = sorted(_dict.items(),key=lambda x:x[0])
		return _list

	def real_one(self,_id):
		try:
			patent = self.params['patent'][str(_id)]
		except KeyError,e:
			return None
		cites = patent['cite']
		_dict = {}
		for i in range(len(cites)):
			year = int(float(cites[i])) + int(float(patent['year']))
			_dict[year] = i + 1
		_list = sorted(_dict.items(),key=lambda x:x[0])
		return _list

	def predict_year_by_year(self,tr,cut_point,duration,spontaneous,w1,alpha,w2):
		N = tr.shape[1] 
		pred = []
		for t in range(cut_point+1,cut_point+duration+1):
			delta_ct = spontaneous/w1*(numpy.exp(-w1*(t-1))-numpy.exp(-w1*t)) + \
				alpha/w2*(numpy.sum(numpy.exp(-w2*((t-1)-tr)))-numpy.sum(numpy.exp(-w2*(t-tr))))
			delta_ct = delta_ct[0,0]
			if len(pred) == 0:
				ct = N + delta_ct
			else :
				ct = pred[-1] + delta_ct
			tr = tr.tolist()[0]
			tr.extend([t for i in range(int(delta_ct))])
			tr = numpy.mat(tr)
			pred.append(ct)
		return pred

	def create_trainable_model(self,sequences, pred_length):
		from keras import backend as K
		from keras.engine.topology import Layer
		from keras.initializers import Constant
		from keras.layers import Input
		from keras.models import Model

		import numpy as np

		import tensorflow as tf
		from tensorflow.python.ops import tensor_array_ops
		from tensorflow.python.ops import control_flow_ops

		class HawkesLayer(Layer):
			def __init__(self, sequences_value, pred_length, delta = 1., **kwargs):
				"""
				can only be the first layer of an architecture
					
				sequences_value[sequence, event, type, feature]

				sequences only contain training events
				"""
				self.sequences_value = np.array(sequences_value,dtype='float32')
				self.sequences_initializer = Constant(self.sequences_value)
				shape = self.sequences_value.shape
				self.nb_sequence = shape[0]
				self.nb_event = shape[1]
				self.nb_type = shape[2]
				self.nb_feature = shape[3]
				self.pred_length = pred_length
				self.delta = delta

				super(HawkesLayer, self).__init__(**kwargs)

			def build(self, input_shape):

				assert len(input_shape) == 2
				assert input_shape[1] == 1 # currenly only support one sample per batch

				self.sequences = self.add_weight(shape=(self.nb_sequence, self.nb_event, self.nb_type, self.nb_feature),
											initializer=self.sequences_initializer,
											trainable=False)

				self.spontaneous = self.add_weight(shape=(self.nb_sequence, self.nb_type),
											initializer='glorot_uniform',
											trainable=True)

				self.Theta = self.add_weight(shape=(self.nb_sequence, self.nb_type),
											initializer='glorot_uniform',
											trainable=True)

				self.W = self.add_weight(shape=(self.nb_sequence, self.nb_type),
											initializer='glorot_uniform',
											trainable=True)

				self.Alpha = self.add_weight(shape=(self.nb_sequence, self.nb_type, self.nb_type),
											initializer='glorot_uniform',
											trainable=True)

				super(HawkesLayer, self).build(input_shape)

			def call(self, seq_id):
				if K.dtype(seq_id) != 'int32':
					seq_id = K.cast(seq_id, 'int32')

				# seq_id = K.gather(seq_id, 0)
				# seq_id = seq_id[0,0]
				seq_id = K.gather(K.gather(seq_id,0),0)
				self.train_seq = K.gather(self.sequences, seq_id)[:,:,0] # currently only support the 1st feature
				spont  = K.gather(self.spontaneous, seq_id)
				theta = K.gather(self.Theta, seq_id)
				w = K.gather(self.W, seq_id)
				alpha = K.gather(self.Alpha, seq_id)
				# print {
				# 	'spont':spont.shape,
				# 	'theta':theta.shape,
				# 	'train_seq':self.train_seq.shape,
				# 	'alpha':alpha,
				# 	'w':w.shape,
				# 	'Theta':self.Theta.get_shape(),
				# 	'Alpha':self.Alpha.get_shape(),
				# 	'sequences':self.sequences.get_shape,
				# 	'seq_id':seq_id.get_shape(),
				# }

				pred_seq = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.nb_event + self.pred_length, 
					dynamic_size=False, infer_shape=True)

				def copy_unit(t, pred_seq):
					pred_seq = pred_seq.write(t, self.train_seq[t])
					return t+1, pred_seq

				def triggering_unit(int_tao, pred_seq, spont, theta, w, alpha, t, effect):
					tao = K.cast(int_tao, 'float32')
					effect_unit = pred_seq.read(int_tao) * (tf.exp(- w * (t - tao) * self.delta) - tf.exp(- w * (t + 1 - tao) * self.delta))
					# print {
					# 	"effect_unit":effect_unit.get_shape(),
					# 	"pred_seq":pred_seq.read(int_tao).get_shape(),
					# 	"0":(tf.exp(- w * (t - tao) * self.delta) - tf.exp(- w * (t + 1 - tao) * self.delta)).get_shape(),
					# }
					return int_tao + 1, pred_seq, spont, theta, w, alpha, t, effect + effect_unit

				def prediction_unit(int_t, pred_seq, spont, theta, w, alpha):
					t = K.cast(int_t, 'float32')
					term1 = spont / theta * (tf.exp(- theta * t * self.delta) - tf.exp(- theta * (t + 1) * self.delta))
					# term2 = tf.stack([pred_seq.read(tao) * (tf.exp(- w * (t - tao) * self.delta) - tf.exp(- w * (t + 1 - tao) * self.delta)) \
					# 			for tao in range(int_t) ])
					# term2 = tf.reduce_sum(term2,0)
					_0, _1, _2, _3, _4, _5, _6, effect = control_flow_ops.while_loop(
						cond=lambda int_tao, _1, _2, _3, _4, _5, _6, _7: int_tao < int_t,
						body=triggering_unit,
						loop_vars=(tf.constant(0, dtype=tf.int32),pred_seq, spont, theta, w, alpha, t, tf.constant([0.,0.],dtype=tf.float32)))

					term2 = tf.reduce_sum(tf.matmul(alpha,tf.expand_dims(effect,1)),1) / w
					pred_seq = pred_seq.write(int_t, term1 + term2)
					return int_t+1, pred_seq, spont, theta, w, alpha

				_0, pred_seq = control_flow_ops.while_loop(
					cond=lambda t, pred_seq: t < self.nb_event,
					body=copy_unit,
					loop_vars=(tf.constant(0, dtype=tf.int32),pred_seq))

				_0, pred_seq, _2, _3, _4, _5 = control_flow_ops.while_loop(
					cond=lambda int_t, _1, _2, _3, _4, _5: int_t < self.nb_event + self.pred_length,
					body=prediction_unit,
					loop_vars=(tf.constant(self.nb_event, dtype=tf.int32),pred_seq,spont,theta,w,alpha))

				pred_seq = tf.expand_dims(tf.expand_dims(pred_seq.stack(), 2), 0)  # currently only support the 1st feature and one sample per batch
				return pred_seq
				

			def compute_output_shape(self, input_shape):
				# print '@1 ',input_shape
				return (input_shape[0], self.nb_event + self.pred_length, self.nb_type, self.nb_feature)


		layer = HawkesLayer(sequences,pred_length)
		x = Input(shape=(1,), dtype='int32')
		y = layer(x)

		model = Model(input=[x], output=[y])

		self.model = model
		return model

	def load(self,f):
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
		train_seq = []
		test_seq = []
		sequences = []
		features = []
		publish_years = []
		for i in range(I):
			publish_year = data[i * 4 + 2][0]
			self_seq = data[i * 4]
			nonself_seq = data[i * 4 + 1]
			feature = data[i * 4 + 3]
			time_seq = self_seq + nonself_seq
			time_seq.sort()

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

		model = Model(input=[f,k2], output=[n1])
		model.compile(optimizer='adam', loss='mape')
		model.fit(
			[numpy.array([[f] for f in features]),numpy.array(sequences)], [numpy.array(labels)],
			nb_epoch=500, batch_size=10,
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
		predictor = HawkesGenerator()
		loaded = predictor.load('../data/paper3.txt')
		model = predictor.pre_train(*loaded)
		# result = predictor.predict(predictor.train(*loaded,max_iter=2),*loaded)
		# print result
		# with open('../data/model/generator.pkl','wb') as f:
		# 	pickle.dump(predictor.train(*loaded),f)
		# with open('../data/model/generator.pkl','rb') as f:
		# 	model = pickle.load(f)
		# 	predictor.params = model
		# result = predictor.predict(model,*loaded)
		# print result

		pass		