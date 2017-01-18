#coding:utf-8
import numpy
import os,sys,time
import json,csv
import cPickle as pickle
numpy.random.seed(1337)

class Single(object):
	def __init__(self):
		self.params = {}

	def train(self,sequences,features,publish_years,pids,threshold,cut=None,predict_year=2000,max_iter=0,max_outer_iter=100):



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
		# W1 = [0.05]*train_count 
		sw1 = 0.05
		alpha = [1.0]*train_count
		# W2 = [1.0]*train_count 
		sw2 = 1.0
		old_tlikelihood = 0.0

		init_time = time.time()
		init_clock = time.clock()
		times = 1
		outer_times = 1
		while 1:
			# update alpha,beta
			old_likelihood = 0
			while 1:
				v1 = numpy.mat([0.0]*num_feature) 
				v2 = 0
				for sam in range(train_count): 
					s = sequences[sam]
					s = [x for x in s if x <= train_times[sam]]
					n = len(s)
					fea = numpy.mat(features[sam])
					# sw1 = numpy.mat(W1[sam])
					# sw2 = numpy.mat(W2[sam])
					salpha = alpha[sam]
					old_obj = 0 

					while 1:
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
						if alpha2 == 0.0:
							print 'error: alpha2 == 0.0'
							print 'alpha1 = ',alpha1
							print 'alpha2 = ',alpha2
							print 'sam = ',sam
							print 'sequences[sam]',sequences[sam]
							print 'T[sam]',T[sam]
							print 's',s
							print '(T[sam] - numpy.array(s)) = ',(T[sam] - numpy.array(s))
							exit()
						salpha = alpha1/float(alpha2)
						if salpha > 1e-2: 
							pass
						else:
							salpha=1e-2
						obj = self.calculate_objective(beta*fea.T,sw1,salpha,sw2,s,train_times[sam])
						if abs(old_obj-obj) < threshold * float(train_count):
							break
						old_obj = obj 
						if obj == -numpy.inf:
							print 'obj = ',obj
							print 'salpha = ',salpha
							print 'sw1 = ',sw1
							print 'sw2 = ',sw2
							print 's = ',s
							print 'beta*fea.T = ',beta*fea.T
							exit()
						# break

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

				# compute likilyhood
				likelihood = 0;
				for item in range(train_count):
					fea = numpy.mat(features[item])
					# sw1 = W1[item]
					# sw2 = W2[item]
					salpha = alpha[item]
					s = sequences[item]
					s = [x for x in s if x <= train_times[item]]
					obj = self.calculate_objective(beta*fea.T,sw1,salpha,sw2,s,train_times[item])
					likelihood = likelihood - obj

				print {
					'iter':times,
					'outer_iter':outer_times,
					'time':time.time() - init_time,
					'clock':time.clock() - init_clock,
					'LL':likelihood[0,0],
					'w1':sw1,
					'w2':sw2,
					'mean_alpha':numpy.mean(alpha),
				}
				times += 1

				if times > max_iter:# or abs(old_likelihood-likelihood) < threshold:
					break
				# old_likelihood = likelihood

			# update w by gradient descent

			for sam in range(train_count):
				s = sequences[sam]
				s = numpy.mat([x for x in s if x <= train_times[sam]])
				n = s.shape[1]
				fea = numpy.mat(features[sam])
				# sw1 = numpy.mat(W1[sam])
				# sw2 = numpy.mat(W2[sam])
				salpha = alpha[sam]
				# old_obj = 0
				# count = 0
				step_size = 1e-3
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
				if sw1 > 1e-5 and sw1 < 1e0:
					pass
				else:
					sw1 = 5e-2
				if sw2 > 1e-3 and sw2 < 1e1:
					pass
				else:
					sw2 = 1
				# obj = self.calculate_objective(beta*fea.T,sw1,salpha,sw2,s.tolist()[0],train_times[sam])
				# if abs((old_obj - obj)/float(train_count)) < threshold or count > 2e1 :
				# 	break
				# old_obj = obj
				# count = count +1
			# W1[sam] = sw1
			# W2[sam] = sw2

			# compute likilyhood
			# tlikelihood = 0
			# for item in range(train_count):
			# 	fea = numpy.mat(features[item])
			# 	# sw1 = W1[item]
			# 	# sw2 = W2[item]
			# 	salpha = alpha[item]
			# 	s = sequences[item]
			# 	s = [x for x in s if x <= train_times[item]]
			# 	obj = self.calculate_objective(beta*fea.T,sw1,salpha,sw2,s,train_times[item])
			# 	tlikelihood = tlikelihood - obj

			if outer_times > max_outer_iter:# or abs(old_tlikelihood-tlikelihood)< threshold/10.0:
				break
			outer_times += 1
			# old_tlikelihood = tlikelihood

		patent = {}
		for item in range(len(alpha)):
			a_patent = {}
			a_patent['alpha'] = alpha[item]
			a_patent['fea'] = features[item]
			a_patent['cite'] = sequences[item]
			a_patent['year'] = publish_years[item]
			patent[pids[item]] = a_patent

		params = {}
		params['predict_year'] = predict_year
		params['train_count'] = len(patent)
		params['beta'] = beta.tolist()[0]
		params['w1'] = sw1
		params['w2'] = sw2
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
		return params

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
		w1 = self.params['w1']
		alpha = patent['alpha']
		w2 = self.params['w2']
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

if __name__ == '__main__':
	predictor = Single()
	loaded = predictor.load('train_sequence_large_long_new.txt')
	# model = predictor.train(*loaded)
	# result = predictor.predict(predictor.train(*loaded,max_iter=2),*loaded)
	# print result
	# with open('tmp.json','wb') as f:
	# 	pickle.dump(predictor.train(*loaded),f)
	with open('tmp.json','rb') as f:
		model = pickle.load(f)
		predictor.params = model
	result = predictor.predict(model,*loaded)
	print result

	pass		