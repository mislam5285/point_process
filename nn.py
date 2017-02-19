#coding:utf-8
import os,sys,csv,numpy

class RNN(object):
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


	def predict(self,model,sequences,labels,features,publish_years,pids,superparams):
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
		superparams['threshold'] = 0.01
		superparams['maxepoch'] = 1000
		superparams['learnrate'] = 1e-3
		superparams['learndecay'] = 1e-8
		superparams['converge'] = 1e-8
		superparams['ncontext'] = 2
		superparams['word_dimension'] = 10
		superparams['hidden_units'] = 100
		superparams['sgd_optimizer'] = 'adam'
		superparams['activation'] = 'relu'
		superparams['dropout'] = 0.5
		superparams['reglambda'] = 1e-5
		return sequences,labels,features,publish_years,pids,superparams

if __name__ == '__main__':
	with open('data/train_log.txt','w') as f:
		sys.stdout = f
		predictor = RNN()
		loaded = predictor.load('data/paper2.txt')
		single_10_result = predictor.predict(predictor.train(*loaded,cut=10,max_outer_iter=0),*loaded)







