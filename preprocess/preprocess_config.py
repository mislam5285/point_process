#coding:utf-8

import os


class PaperConfig(object):
	def __init__(self):
		self.params_file = os.path.dirname(os.path.abspath(__file__)) + '/../service/params.json'
		#params_file = os.path.dirname(os.path.abspath(__file__)) + '/' + 'service/param15.json'

		# original dataset
		self.dataset_dir = '/Volumes/exFAT/tmp/academic'
		self.preprocessed_dataset_dir = os.path.dirname(os.path.abspath(__file__)) + '/../dataset/academic'
		self.aln_seq = self.preprocessed_dataset_dir + '/aln_seq.csv'
		self.aln_fea = self.preprocessed_dataset_dir + '/aln_fea.csv'
		self.aln_vau = self.preprocessed_dataset_dir + '/aln_vau.csv'


paper_config = PaperConfig()