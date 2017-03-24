#coding:utf-8
import os,sys,csv,json
import numpy
import numpy as np
import multiprocessing

from generator import HawkesGenerator
from discriminator import CNNDiscriminator

cpu = multiprocessing.cpu_count()

class HawkesGAN(object):
	def __init__(self):
		self.gen = HawkesGenerator()
		self.dis = CNNDiscriminator()

	def full_train(self,full_sequences,train_sequences,features,publish_years,pids,superparams):
		from keras.layers import Input
		from keras.models import Model
		from keras.optimizers import SGD
		from customed_layer import PoissonNoise

		nb_sequence = len(full_sequences)
		nb_event = len(train_sequences[0])
		pred_length = len(full_sequences[0]) - nb_event
		nb_type = len(full_sequences[0][0])
		nb_feature = len(full_sequences[0][0][0])

		# guarantee pretrained generator
		if self.gen.sequence_weights is None:
			raise Exception('generator not pretrained, or weights not loaded')

		# build keras models
		self.gen.create_trainable_model(train_sequences,pred_length)
		# self.gen.model.compile(optimizer='adam', loss='mape', metrics=['accuracy'])
		# self.gen.model.fit(np.arange(nb_sequence),np.array(full_sequences),verbose=1,batch_size=1,epochs=100)
		self.dis.create_trainable_model(nb_event + pred_length,nb_type,nb_feature)
		self.dis.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

		self.dis.model.trainable = False
		for l in self.dis.model.layers: l.trainable = False
		z = Input(batch_shape=(1,1), dtype='int32')
		g_z = self.gen.model(z)
		noised_g_z = PoissonNoise(train_sequences,pred_length)(g_z)
		y = self.dis.model(g_z)
		noised_gen_model = Model(inputs=[z], outputs=[noised_g_z])
		gan_model = Model(inputs=[z], outputs=[y])
		gan_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

		# pretrain discrimnator
		Z = np.arange(nb_sequence)
		X = np.array(full_sequences)
		b_size = 512
		max_pretrain_iter = 5
		it = 0

		likelihood = self.compute_likelihood()
		mape_acc = self.compute_mape_acc(self.gen.model,Z,X,pred_length)
		print {
			'source':'before pre-training discriminator',
			'mape':mape_acc['mape'],
			'acc':mape_acc['acc'],
			'LL':likelihood,
		}
		sys.stdout.flush()

		while it <= max_pretrain_iter * b_size:
			batch_z = Z[np.arange(it,it+b_size)%nb_sequence]
			batch_g_z = noised_gen_model.predict(batch_z,batch_size=1)
			batch_x = X[np.arange(it,it+b_size)%nb_sequence]
			batch_x_merge = np.array([batch_g_z[i/2] if i % 2 == 0 else batch_x[i/2] for i in range(2 * b_size)])
			batch_y_merge = np.array([[0.,1.] if i % 2 == 0 else [1.,0.] for i in range(2 * b_size)])
			history = self.dis.model.fit(batch_x_merge,batch_y_merge,batch_size=1,epochs=1,verbose=1)

			it += b_size

			# print history.history
			# exit()

			# if it + 1 >= max_pretrain_iter: break


		# full train gan model
		Z = np.arange(nb_sequence)
		X = np.array(full_sequences)
		Y = np.array([[1.,0.] for i in range(nb_sequence)])
		b_size_dis = 32
		b_size_gan = 512
		it_dis = 0
		it_gan = 0
		max_fulltrain_iter = 500

		# for epoch in range(500): # use while , different batch size for G and D
		# for it in range(0,nb_sequence,b_size):
		while it_gan <= b_size_gan * max_fulltrain_iter:
			batch_z = Z[np.arange(it_dis,it_dis+b_size_dis)%nb_sequence]
			batch_g_z = noised_gen_model.predict(batch_z,batch_size=1)
			batch_x = X[np.arange(it_dis,it_dis+b_size_dis)%nb_sequence]
			batch_x_merge = np.array([batch_g_z[i/2] if i % 2 == 0 else batch_x[i/2] for i in range(2 * b_size_dis)])
			batch_y_merge = np.array([[0.,1.] if i % 2 == 0 else [1.,0.] for i in range(2 * b_size_dis)])
			dis_his = self.dis.model.fit(batch_x_merge,batch_y_merge,batch_size=1,epochs=1,verbose=0)

			batch_z_gan = Z[np.arange(it_gan,it_gan+b_size_gan)%nb_sequence]
			batch_y_gan = np.array([[1.,0.] for i in range(b_size_gan)])
			gan_his = gan_model.fit(batch_z_gan,batch_y_gan,batch_size=1,epochs=1,verbose=1)

			likelihood = self.compute_likelihood()
			mape_acc = self.compute_mape_acc(self.gen.model,Z,X,pred_length)
			print {
				'source':'full train one batch',
				# 'epoch':epoch,
				# 'iteration':it,
				# 'batch':'[' + str(it) + ',' + str(it+b_size) + ')',
				'dis_loss':dis_his.history['loss'],
				'gan_loss':gan_his.history['loss'],
				'mape':mape_acc['mape'],
				'acc':mape_acc['acc'],
				'LL':likelihood,
			}
			sys.stdout.flush()

			it_dis += b_size_dis
			it_gan += b_size_gan


		self.gen.model.summary()
		self.dis.model.summary()
		gan_model.summary()

	def compute_likelihood(self):
		return 0.0

	def compute_mape_acc(self,generator,z,y,pred_length):
		g_z = generator.predict(z,batch_size=1)
		assert g_z.shape == y.shape
		count_g_z = np.sum(np.sum(g_z,3),2)
		count_y = np.sum(np.sum(y,3),2)
		for i in range(1,g_z.shape[1]):
			count_g_z[:,i] += count_g_z[:,i-1]
			count_y[:,i] += count_y[:,i-1]
		count_g_z = count_g_z[:,-pred_length:]
		count_y = count_y[:,-pred_length:]
		mape = np.mean(np.abs(count_g_z - count_y)/(count_y + 0.1),0)
		acc = np.mean(np.abs(count_g_z - count_y)/(count_y + 0.1) < 0.3,0)
		return {'mape':mape.tolist(),'acc':acc.tolist()}


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
				# sequence.append([[self_count],[nonself_count]])
				sequence.append([[self_count + nonself_count]])
			full_sequences.append(sequence)

			features.append(feature)
			publish_years.append(publish_year)

			sequence = []
			for year in range(train_length) :
				self_count = len([x for x in self_seq if year <= x and x < year + 1])
				nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1])
				# sequence.append([[self_count],[nonself_count]])
			# for year in range(train_length,int(pred_length + train_length)) :
			# 	self_count = len([x for x in self_seq if year <= x and x < year + 1]) * 1.5
			# 	nonself_count = len([x for x in nonself_seq if year <= x and x < year + 1]) * 1.5
				# sequence.append([[self_count],[nonself_count]])
				sequence.append([[self_count + nonself_count]])
			train_sequences.append(sequence)

		superparams = {}
		# print {
		# 	# 'np.mean(np.array(full_sequences)[:,15:])':np.mean(np.array(full_sequences)[:,15:]),
		# 	'np.mean(np.array(full_sequences)[:,0:15])':np.mean(np.array(full_sequences)[:,0:15]),
		# 	'np.mean(np.array(train_sequences)[:,0:15])':np.mean(np.array(train_sequences)[:,0:15]),
		# }
		# exit()
		return full_sequences,train_sequences,features,publish_years,pids,superparams


# yield
if __name__ == '__main__':
	with open('../log/train_gan.log','w') as f:
		old_stdout = sys.stdout
		sys.stdout = f
		gan = HawkesGAN()
		try:
			gan.gen.sequence_weights = json.load(open('../data/paper.3.pretrain.sequence_weights.json'))
		except:
			loaded = gan.gen.load('../data/paper3.txt')
			gan.gen.pre_train(*loaded,max_outer_iter=1)
			with open('../data/paper.3.pretrain.sequence_weights.json','w') as fw:
				json.dump(gan.gen.sequence_weights,fw)
		# exit()
		loaded = gan.load('../data/paper3.txt')
		gan.full_train(*loaded)
		sys.stdout = old_stdout