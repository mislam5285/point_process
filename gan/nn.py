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
		predictor = RNN()
		loaded = predictor.load('../data/paper2.txt')
		predictor.train(*loaded,cut=10,max_outer_iter=0)







