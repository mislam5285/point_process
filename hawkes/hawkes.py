#coding:utf-8
import numpy
import os,sys,time
import json,csv
numpy.random.seed(1337)

class MHawkes(object):
	def __init__(self):
		pass

	def train(self, train_seq, test_seq, superparams, initparams,max_iter=0,max_outer_iter=100):

		model = {}
		model['superparams'] = superparams
		I = len(train_seq)
		M = 2
		sigma = superparams['sigma']
		model['mu'] = numpy.random.random(M)
		model['AA'] = numpy.random.random(M**2)
		model['theta'] = initparams['theta']
		model['w'] = initparams['w']
		model['M'] = M
		model['I'] = I

		reg_alpha = numpy.zeros(model['AA'].shape)

		init_time = time.time()
		init_clock = time.clock()
		iteration = 1
		outer_times = 1

		L_converge = False
		while not L_converge:
			Q_converge = False
			while not Q_converge:
				LL = 0

				Bmu = numpy.zeros(model['mu'].shape)
				Cmu = 0.0

				Gamma = numpy.random.random((M**2,I))
				for i in range(I):
					T = train_seq[i]['times'][-1] + 0.01
					times = numpy.array(train_seq[i]['times'],dtype=float)
					dims = numpy.array(train_seq[i]['dims'],dtype=int)
					N = len(times)
					Cmu += T

					B = sigma * reg_alpha
					B = B.reshape((M,M))
					C = numpy.zeros(B.shape)
					AA = model['AA'].reshape(M,M)
					w = model['w']

					LL += T * numpy.sum(model['mu'])

					for j in range(0,N):
						t_j = times[j]
						m_j = dims[j]
						int_g = self.G(w,T - times[j])
						B[:,m_j] += int_g
						LL += numpy.sum(int_g * AA[:,m_j])
						_lambda = model['mu'][m_j]
						if j == 0:
							psi = 1.0
						else:
							psi = _lambda
							t_k = times[0:j]
							m_k = dims[0:j]
							idx = [_idx for _idx in range(len(t_k)) if t_j - t_k[_idx] <= superparams['impact_period']]
							if len(idx) == 0:
								psi /= _lambda
							else:
								_g = self.g(w,t_j - t_k[idx])
								_a = AA[m_j, m_k[idx]]
								phi = _g * _a
								_lambda += numpy.sum(phi)
								psi /= _lambda
								phi /= _lambda

								for k in range(len(idx)):
									C[m_j,m_k[idx[k]]] += phi[k]
							LL -= numpy.log(_lambda)
						Bmu[m_j] += psi
					AA = (- B + numpy.sqrt(B ** 2 + 8 * sigma * C )) / (4 * sigma) * numpy.sign(C)
					Gamma[:,i] = AA.reshape(M**2)

				mu = Bmu / Cmu
				model['mu'] = mu

				# check convergence
				AA = numpy.mean(Gamma,1)
				error = numpy.sum(numpy.abs(model['AA'] - AA)) / numpy.sum(model['AA'])
				# print json.dumps({'iter':iteration,'time':time.time() - init_time,'clock':time.clock() - init_clock,'LL':LL,'error':error})
				print json.dumps({
					'iter':iteration,
					'outer_iter':outer_times,
					'time':time.time() - init_time,
					'clock':time.clock() - init_clock,
					'LL':LL,
					'error':error,
					'mean_alpha':AA.tolist(),
					# 'mean_alpha_self':numpy.mean(Gamma),
					# 'mean_alpha_nonself':numpy.mean(Gamma),
				})
				iteration += 1

				model['AA'] = AA
				if iteration > max_iter or error < superparams['thres']:
					Q_converge = True
					break
				else:
					Q_converge = False


			if outer_times > max_outer_iter:# or abs(old_tlikelihood-tlikelihood)< threshold/10.0:
				L_converge = True
			else:
				L_converge = False
			outer_times += 1
		return model

	def G(self,w,t):
		return (1 - numpy.exp(- w * t)) / w

	def g(self,w,t):
		return numpy.exp(- w * t)

	def predict(self, model, train_seq, test_seq, superparams, initparams):
		M = model['M']
		pred_seqs = []
		pred_seqs_self = []
		pred_seqs_nonself = []

		real_seqs = []
		real_seqs_self = []
		real_seqs_nonself = []
		mapes = []

		mapes_self = []
		mapes_nonself = []
		I = len(train_seq)

		for i in range(I):
			T = train_seq[i]['times'][-1] + 0.01
			times = numpy.array(train_seq[i]['times'],dtype=float)
			dims = numpy.array(train_seq[i]['dims'],dtype=int)
			features = numpy.array(train_seq[i]['features'],dtype=float)
			N = len(times)
			N_self = len([x for x in dims if x == 0])
			N_nonself = len([x for x in dims if x == 1])
			if N != N_self + N_nonself:
				print 'N != N_self + N_nonself'
				exit()

			AA = model['AA'].reshape(M,M)
			w = model['w']

			duration = model['superparams']['duration']
			mape = []
			mape_self = []
			mape_nonself = []

			pred_seq = []
			pred_seq_self = []
			pred_seq_nonself = []

			real_seq = []
			real_seq_self = []
			real_seq_nonself = []

			for year in range(duration+1):
				LL = 0
				LL_self = 0
				LL_nonself = 0
				for j in range(N):
					m_j = dims[j]
					int_g = self.G(w,T + year - times[j]) - self.G(w,T - times[j])
					LL += numpy.sum(int_g * AA[:,m_j])
					LL_self += int_g * AA[0,m_j]
					LL_nonself += int_g * AA[1,m_j]

				LL += year * numpy.sum(model['mu'])
				LL_self += year * numpy.sum(model['mu'])
				LL_nonself += year * numpy.sum(model['mu'])


				pred = N + LL
				pred_self = N_self + LL_self
				pred_nonself = N_nonself + LL_nonself
				real = N + len([x for x in test_seq[i]['times'] if x + model['superparams']['cut_point'] < T + year])
				real_self = N_self + len([x for _x,x in enumerate(test_seq[i]['times']) if x + model['superparams']['cut_point'] < T + year and test_seq[i]['dims'][_x] == 0])
				real_nonself = N_nonself + len([x for _x,x in enumerate(test_seq[i]['times']) if x + model['superparams']['cut_point'] < T + year and test_seq[i]['dims'][_x] == 1])
				mape.append(abs(pred - real) / float(real + 0.001))
				mape_self.append(abs(pred_self - real_self) / float(real_self + 0.001))
				mape_nonself.append(abs(pred_nonself - real_nonself) / float(real_nonself + 0.001))

				pred_seq.append(pred)
				pred_seq_self.append(pred_self)
				pred_seq_nonself.append(pred_nonself)

				real_seq.append(real)
				real_seq_self.append(real_self)
				real_seq_nonself.append(real_nonself)

			mapes.append(mape)
			mapes_self.append(mape_self)
			mapes_nonself.append(mape_nonself)

			pred_seqs.append(pred_seq)
			pred_seqs_self.append(pred_seq_self)
			pred_seqs_nonself.append(pred_seq_nonself)

			real_seqs.append(real_seq)
			real_seqs_self.append(real_seq_self)
			real_seqs_nonself.append(real_seq_nonself)

		av_mape = numpy.mean(numpy.array(mapes),0)
		av_acc = numpy.mean(numpy.array(mapes) < model['superparams']['epsilon'],0)

		av_mape_self = numpy.mean(numpy.array(mapes_self),0)
		av_acc_self = numpy.mean(numpy.array(mapes_self) < model['superparams']['epsilon'],0)

		av_mape_nonself = numpy.mean(numpy.array(mapes_nonself),0)
		av_acc_nonself = numpy.mean(numpy.array(mapes_nonself) < model['superparams']['epsilon'],0)

		return {
			'av_mape':av_mape.tolist(),
			'av_acc':av_acc.tolist(),
			'av_mape_self':av_mape_self.tolist(),
			'av_acc_self':av_acc_self.tolist(),
			'av_mape_nonself':av_mape_nonself.tolist(),
			'av_acc_nonself':av_acc_nonself.tolist(),
			# 'mapes':mapes,
			# 'mapes_self':mapes_self,
			# 'mapes_nonself':mapes_nonself,
			# 'pred_seqs':pred_seqs,
			# 'pred_seqs_self':pred_seqs_self,
			# 'pred_seqs_nonself':pred_seqs_nonself,
			# 'real_seqs':real_seqs,
			# 'real_seqs_self':real_seqs_self,
			# 'real_seqs_nonself':real_seqs_nonself,
			}

	def predict_one(self,patent_id):
		pass

	def load(self,f,cut=15):
		data = []
		for i,row in enumerate(csv.reader(file(f,'r'))):
			if i % 4 == 2:
				row = [float(row[1])]
			elif i % 4 == 0 or i % 4 == 1:
				row = [float(x) for x in row[1:]]
			elif i % 4 == 3:
				_row = [float(x) for x in row[1:]]
				_max = max(_row)
				_min = min(_row)
				row = [(x - _min)/float(_max - _min) for x in _row]
			data.append(row)

		T = cut
		lines = 4
		I = int(len(data)/lines)
		train_seq = []
		test_seq = []
		for i in range(I):
			publish_year = data[i * lines + 2]
			self_seq = data[i * lines]
			nonself_seq = data[i * lines + 1]
			feature = data[i * lines + 3]
			varying = data[(i * lines + 4):(i * lines + lines)]

			time_seq = self_seq + nonself_seq
			dim_seq = ([0] * len(self_seq)) + ([1] * len(nonself_seq))
			S = zip(time_seq,dim_seq)
			S = sorted(S,key=lambda x:x[0])
			Y = [x[0] for x in S]
			dim_seq = [x[1] for x in S]
			cut_point = T
			time_train = [y for y in Y if y <= cut_point]
			dim_train = [e for i,e in enumerate(dim_seq) if Y[i] <= cut_point]
			if len(time_train) < 5:
				continue
			_dict = {}
			_dict['times'] = time_train
			_dict['dims'] = dim_train
			_dict['features'] = feature
			_dict['varying'] = varying
			_dict['publish_year'] = publish_year
			train_seq.append(_dict)

			_dict = {}
			_dict['times'] = [y - cut_point for y in Y if y > cut_point]
			_dict['dims'] = [e for i,e in enumerate(dim_seq) if Y[i] > cut_point]
			_dict['features'] = feature
			_dict['varying'] = varying
			_dict['publish_year'] = publish_year
			test_seq.append(_dict)


		superparams = {}
		superparams['M'] = 2
		superparams['outer'] = 20
		superparams['inner'] = 10
		superparams['K'] = len(train_seq[0]['features'])
		superparams['impact_period'] = 5
		superparams['sigma'] = 1
		superparams['thres'] = 1e-3
		superparams['cut_point'] = cut_point
		superparams['duration'] = 10
		superparams['epsilon'] = 0.3

		initparams = {}
		initparams['theta'] = 0.2
		initparams['w'] = 1.0
		initparams['b'] = 0.5


		return train_seq,test_seq,superparams,initparams


if __name__ == '__main__':
	predictor = MHawkes()
	loaded = predictor.load('train_sequence_large_long_new.txt')
	# model = predictor.train(*loaded)
	result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
	# print loaded
	# print result['av_mape']
	# print result['av_acc']
	print result
	pass
