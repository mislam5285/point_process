#coding:utf-8
import os,sys,csv
import numpy as np
import multiprocessing

from gan.generator import HawkesGenerator
from gan.discriminator import CNNDiscriminator

cpu = multiprocessing.cpu_count()

class HawkesGAN(object):
	def __init__(self):
		self.gen = HawkesGenerator()
		self.dis = CNNDiscriminator()

	def full_train(self,full_sequences,train_sequences,features,publish_years,pids,superparams):
		from keras.layers import Input, Dense, Flatten, Convolution2D, Activation, Dropout, merge
		from keras.models import Model
		from keras.regularizers import l1,l2

		nb_sequence = len(full_sequences)
		nb_event = len(train_sequences[0])
		pred_length = len(full_sequences[0]) - nb_event
		nb_feature = len(full_sequences[0][0])

		self.gen.create_trainable_model(train_sequences,pred_length)
		self.gen.model.summary()

		self.dis.create_trainable_model(nb_event + pred_length,nb_type,nb_feature)
		self.dis.model.summary()

		self.dis.model.trainable = False
		for l in self.dis.model.layers: l.trainable = False

		x = Input(shape=(1,), dtype='int32')
		y = self.gen.model(x)
		z = self.dis.model(y)

		gan = Model(input=[x], output=[z])
		gan.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		gan.summary()


	def pre_train(self,paper_data):
		loaded = self.gen.load(paper_data)
		self.gen.pre_train(*loaded,max_outer_iter=10)

	def load(self,f,pred_length=10,train_length=15):
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
		full_sequences = []
		train_sequences = []
		features = []
		publish_years = []
		for i in range(I):
			publish_year = data[i * 4 + 2][0]
			self_seq = data[i * 4]
			nonself_seq = data[i * 4 + 1]
			feature = data[i * 4 + 3]
			#time_seq = self_seq + nonself_seq
			#time_seq.sort()

			sequence = []
			for year in range(int(pred_length + train_length)) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1])
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
				sequence.append([[self_count],[nonself_count]])
			full_sequences.append(sequence)

			features.append(feature)
			publish_years.append(publish_year)

		for i in range(I):
			sequence = []
			for year in range(train_length) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1])
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
				sequence.append([[self_count],[nonself_count]])
			for year in range(train_length,int(pred_length + train_length)) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1]) * 1.5
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1]) * 1.5
				sequence.append([[self_count],[nonself_count]])
			train_sequences.append(sequence)

		superparams = {}
		return full_sequences,train_sequences,features,publish_years,pids,superparams


# yield
if __name__ == '__main__':
	with open('../log/train_gan.log','w') as f:
		pass