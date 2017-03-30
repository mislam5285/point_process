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

from hawkes.mtpp import MTPP
from hawkes.hawkes import MHawkes
from hawkes.single import Single
from point_process.generator import HawkesGenerator
from point_process.gan import HawkesGAN
from preprocess.screen import PaperScreenor

np.random.seed(137)

import os, sys
root = os.path.abspath(os.path.dirname(__file__))

def paper_fix_train_total_xiao():
    will_screen = False
    will_train_mtpp = False
    will_train_single = False
    will_draw = True
    # preprocess
    paper_data = root + '/data/paper3.txt'
    if will_screen == True :
        paper_data_raw = root + '/data/paper2.txt'
        processor = PaperScreenor()
        result = processor.screen(paper_data_raw)
        with open(paper_data,'w') as fw:
            fw.writelines(result)
    
    # training
    mape_acc_data = root + '/data/paper.mape_acc.txt'
    if will_train_mtpp == True :
        predictor = MTPP()
        loaded = predictor.load(paper_data,cut=10)
        proposed_10_result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
        loaded = predictor.load(paper_data,cut=15)
        proposed_15_result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
        loaded = predictor.load(paper_data,cut=20)
        proposed_20_result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
        with open(mape_acc_data) as fr :
            result = json.load(fr)
            result['contrast_single']['10_mape']['proposed'] = proposed_10_result['av_mape']
            result['contrast_single']['10_acc']['proposed'] = proposed_10_result['av_acc']
            result['contrast_single']['15_mape']['proposed'] = proposed_15_result['av_mape']
            result['contrast_single']['15_acc']['proposed'] = proposed_15_result['av_acc']
            result['contrast_single']['20_mape']['proposed'] = proposed_20_result['av_mape']
            result['contrast_single']['20_acc']['proposed'] = proposed_20_result['av_acc']
        with open(mape_acc_data,'w') as fw :
            json.dump(result,fw)

    if will_train_single == True :
        predictor = Single()
        loaded = predictor.load(paper_data)
        single_10_result = predictor.predict(predictor.train(*loaded,cut=10,max_outer_iter=0),*loaded)
        single_15_result = predictor.predict(predictor.train(*loaded,cut=15,max_outer_iter=0),*loaded)
        single_20_result = predictor.predict(predictor.train(*loaded,cut=20,max_outer_iter=0),*loaded)
        with open(mape_acc_data) as fr :
            result = json.load(fr)
            result['contrast_single']['10_mape']['hawkes_decay'] = single_10_result['mape']
            result['contrast_single']['10_acc']['hawkes_decay'] = single_10_result['acc']
            result['contrast_single']['15_mape']['hawkes_decay'] = single_15_result['mape']
            result['contrast_single']['15_acc']['hawkes_decay'] = single_15_result['acc']
            result['contrast_single']['20_mape']['hawkes_decay'] = single_20_result['mape']
            result['contrast_single']['20_acc']['hawkes_decay'] = single_20_result['acc']
        with open(mape_acc_data,'w') as fw :
            json.dump(result,fw)

    # drawing
    if will_draw == True :
        with open(mape_acc_data) as f:
            graph = json.load(f)
            subgraphs = [
                            graph['contrast_single']['10_mape'],
                            graph['contrast_single']['15_mape'],
                            graph['contrast_single']['20_mape'],
                            graph['contrast_single']['10_acc'],
                            graph['contrast_single']['15_acc'],
                            graph['contrast_single']['20_acc'],
                        ]
        colors = ['red','blue','green']
        titles = ['Proposed','Xiao et al.']
        keys = ['proposed','hawkes_decay']
        years = ['(10 years)','(15 years)','(20 years)']
        line_type = ['-','--']
        # plt.subplot(221)

        for i in [0,3]: # mape or acc
            plt.figure()
            for j in [0,1,2]: # train year
                for k in [0,1]: # proposed or xiao 
                    _curve = subgraphs[i + j][keys[k]]
                    y = np.array([float(e) for e in _curve])
                    plt.plot(np.arange(1,len(y)+1),y,line_type[k],c=colors[j],lw=2,label=titles[k] + years[j])
                    plt.scatter(np.arange(1,len(y)+1),y,c=colors[j],lw=0) 

            if i == 0:
                plt.xlabel('')
                plt.title('MPAE')
                plt.legend(loc='lower right')
            if i == 3:
                plt.xlabel('')
                plt.title('ACC')
                plt.legend(loc='lower left')

            plt.gcf().set_size_inches(5., 5., forward=True)
            #plt.show()    
            if i == 0: key = 'paper.fix-train.total.xiao.mape.png'
            if i == 3: key = 'paper.fix-train.total.xiao.acc.png'
            plt.savefig(root + '/pic/%s'%key)
        
def paper_fix_train_non_self_m_hawkes():
    will_screen = False
    will_train_mtpp = False
    will_train_hawkes = False
    will_draw = True
    # preprocess
    paper_data = root + '/data/paper3.txt'
    if will_screen == True :
        paper_data_raw = root + '/data/paper2.txt'
        processor = PaperScreenor()
        result = processor.screen(paper_data_raw)
        with open(paper_data,'w') as fw:
            fw.writelines(result)
    
    # training
    mape_acc_data = root + '/data/paper.mape_acc.txt'
    if will_train_mtpp == True :
        predictor = MTPP()
        loaded = predictor.load(paper_data,cut=10)
        proposed_10_result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
        with open(mape_acc_data) as fr :
            result = json.load(fr)
            result['self_nonself']['mape']['proposed-total'] = proposed_10_result['av_mape']
            result['self_nonself']['mape']['proposed-self'] = proposed_10_result['av_mape_self']
            result['self_nonself']['mape']['proposed-nonself'] = proposed_10_result['av_mape_nonself']
            result['self_nonself']['acc']['proposed-total'] = proposed_10_result['av_acc']
            result['self_nonself']['acc']['proposed-self'] = proposed_10_result['av_acc_self']
            result['self_nonself']['acc']['proposed-nonself'] = proposed_10_result['av_acc_nonself']
        with open(mape_acc_data,'w') as fw :
            json.dump(result,fw)

    if will_train_hawkes == True :
        predictor = MHawkes()
        loaded = predictor.load(paper_data,cut=10)
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
                    plt.plot(np.arange(1,len(y)+1),y,line_type[k],c=colors[j],lw=2,label=model_name[k] + "(" + titles[j] + ")")
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

            plt.gcf().set_size_inches(5., 5., forward=True)
            # plt.show()
            if i == 0: key = 'paper.fix-train.non-self.m-hawks.mape.png'
            if i == 1: key = 'paper.fix-train.non-self.m-hawks.acc.png'
            plt.savefig(root + '/pic/%s'%key)

def paper_hawkes_generator_pretrain_convergence():
    will_screen = False
    will_train_hawkes = False
    will_draw = True
    # preprocess
    paper_data = root + '/data/paper3.txt'
    if will_screen == True :
        paper_data_raw = root + '/data/paper2.txt'
        processor = PaperScreenor()
        result = processor.screen(paper_data_raw)
        with open(paper_data,'w') as fw:
            fw.writelines(result)
    
    # training
    pre_train_log = root + '/data/paper.pretrain.log5.txt'

    if will_train_hawkes == True :
        with open(pre_train_log,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            predictor = HawkesGenerator()
            loaded = predictor.load(paper_data)
            model = predictor.pre_train(*loaded,max_outer_iter=10)
            sys.stdout = old_stdout

    # drawing
    if will_draw == True :
        # plt.figure(figsize=(8,6), dpi=72, facecolor="white")
        colors = ['red','green','purple']
        keys = [lambda x:x['LL'], lambda x:x['acc'][-1], lambda x:x['mape'][-1]]
        labels = ['Loss on observed seq.', 'ACC on test seq.', 'MAPE on test seq.']

        with open(pre_train_log) as f:
            nodes = []
            for line in f:
                try:
                    # node = json.loads(line)
                    node = eval(line)
                    nodes.append(node)
                except:
                    print 'error'

            for i in range(3):
                plt.figure()
                x_right_limit = 200
                y = np.array([float(keys[i](node)) for node in nodes])[0:x_right_limit]
                delta = np.max(y) - np.min(y)
                delta /= 30
                if y[0] > y[-1]: plt.ylim(np.min(y) - delta, 0.2 * np.max(y) + 0.8 * np.min(y))
                if y[0] < y[-1]: plt.ylim(0.8 * np.max(y) + 0.2 * np.min(y), np.max(y) + delta)
                plt.xlim(0,x_right_limit)
                plt.plot(np.arange(1,len(y)+1),y,c=colors[i],lw=2,label=labels[i])
                gca = plt.gca()

                plt.xlabel('iterations')
                plt.title('learning curve')
                plt.legend(loc='upper right')
                plt.gcf().set_size_inches(5., 5., forward=True)

                #plt.show()
                if i == 0: key = 'paper.gan.pretrain.learning.NLL.png'
                if i == 1: key = 'paper.gan.pretrain.learning.ACC.png'
                if i == 2: key = 'paper.gan.pretrain.learning.MAPE.png'
                plt.savefig(root + '/pic/%s'%key)

def paper_full_train_potential_ability():
    will_screen = False
    will_train_hawkes = False
    will_train_mse = False
    will_train_gan = False
    will_train_wgan = False
    will_draw = True
    # preprocess
    paper_data = root + '/data/paper3.txt'
    if will_screen == True :
        paper_data_raw = root + '/data/paper2.txt'
        processor = PaperScreenor()
        result = processor.screen(paper_data_raw)
        with open(paper_data,'w') as fw:
            fw.writelines(result)

    # pre-training
    pre_train_log = root + '/data/paper.pretrain.log5.txt'

    if will_train_hawkes == True :
        with open(pre_train_log,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            predictor = HawkesGenerator()
            loaded = predictor.load(paper_data)
            model = predictor.pre_train(*loaded,max_outer_iter=10)
            sys.stdout = old_stdout

    # full-training
    full_train_mse_log = root + '/data/paper.fulltrain.mse.log.txt'
    pretrain_iter = 1
    alpha_iter=3
    w_iter=30
    full_train_start = pretrain_iter * (alpha_iter + w_iter)

    if will_train_mse == True :
        mse_weight = 1.
        gan_weight = 0.
        with open(full_train_mse_log,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(open(root + '/data/paper.3.pretrain.sequence_weights.json'))
            except:
                loaded = gan.gen.load(paper_data)
                gan.gen.pre_train(*loaded,max_outer_iter=pretrain_iter)
                with open(root + '/data/paper.3.pretrain.sequence_weights.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(paper_data)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight)
            sys.stdout = old_stdout

    full_train_gan_log = root + '/data/paper.fulltrain.gan.log.txt'
    if will_train_gan == True:
        mse_weight = 0.
        gan_weight = 1.
        train_gan_method = 'gan'
        with open(full_train_gan_log,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(open(root + '/data/paper.3.pretrain.sequence_weights.json'))
            except:
                loaded = gan.gen.load(paper_data)
                gan.gen.pre_train(*loaded,max_outer_iter=pretrain_iter)
                with open(root + '/data/paper.3.pretrain.sequence_weights.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(paper_data)
            gan.full_train(*loaded,max_fulltrain_iter=400,train_gan_method=train_gan_method,mse_weight=mse_weight,gan_weight=gan_weight)
            sys.stdout = old_stdout

    full_train_wgan_log = root + '/data/paper.fulltrain.wgan.log.txt'
    if will_train_wgan == True:
        mse_weight = 0.
        gan_weight = 1.
        train_gan_method = 'wgan'
        with open(full_train_wgan_log,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(open(root + '/data/paper.3.pretrain.sequence_weights.json'))
            except:
                loaded = gan.gen.load(paper_data)
                gan.gen.pre_train(*loaded,max_outer_iter=pretrain_iter)
                with open(root + '/data/paper.3.pretrain.sequence_weights.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(paper_data)
            gan.full_train(*loaded,max_fulltrain_iter=400,train_gan_method=train_gan_method,mse_weight=mse_weight,gan_weight=gan_weight)
            sys.stdout = old_stdout



    # drawing
    if will_draw == True :
        # plt.figure(figsize=(8,6), dpi=72, facecolor="white")
        colors = ['red','green','purple']
        keys = [lambda x:x['acc'][-1], lambda x:x['mape'][-1]]
        labels = ['ACC on test seq.', 'MAPE on test seq.']

        f_pretrain = open(pre_train_log)
        f_full_mse = open(full_train_mse_log)
        f_full_gan = open(full_train_gan_log)
        f_full_wgan = open(full_train_wgan_log)
        nodes_pretrain = []
        nodes_full_mse = []
        nodes_full_gan = []
        nodes_full_wgan = []

        for i in range(len(keys)):
            plt.figure()

            for line in f_pretrain:
                try:
                    node = eval(line)
                    nodes_pretrain.append(node)
                except:
                    print 'error'

            for line in f_full_mse:
                try:
                    node = eval(line)
                    nodes_full_mse.append(node)
                except:
                    print 'error'

            for line in f_full_gan:
                try:
                    node = eval(line)
                    nodes_full_gan.append(node)
                except:
                    print 'error'

            for line in f_full_wgan:
                try:
                    node = eval(line)
                    nodes_full_wgan.append(node)
                except:
                    print 'error'

            # arrange layout
            y = np.array([float(keys[i](node)) for node in nodes_pretrain])
            y_full_mse = np.array([float(keys[i](node)) for node in nodes_full_mse])
            y_full_gan = np.array([float(keys[i](node)) for node in nodes_full_gan])
            y_full_wgan = np.array([float(keys[i](node)) for node in nodes_full_wgan])

            delta = max(np.max(y),np.max(y_full_mse)) - min(np.min(y),np.min(y_full_mse))
            delta /= 30.
            x_left_limit = 0
            x_right_limit = 300
            if y[0] > y[-1]:
                y_lower_limit = min(np.min(y),np.min(y_full_mse)) - delta
                y_upper_limit = 0.2 * np.max(y) + 0.8 * np.min(y)
            else:
                y_lower_limit = 0.8 * np.max(y) + 0.2 * np.min(y)
                y_upper_limit = max(np.max(y),np.max(y_full_mse)) + delta

            plt.ylim(y_lower_limit, y_upper_limit)
            plt.xlim(0,x_right_limit)

            # draw curve
            plt.plot(np.arange(1,len(y)+1),y,c=colors[i],lw=2,label=labels[i])
            plt.plot(np.arange(full_train_start,len(y_full_mse)+full_train_start),y_full_mse,c=colors[i],lw=2,label=labels[i])
            plt.plot(np.arange(full_train_start,len(y_full_gan)+full_train_start),y_full_gan,c=colors[i],lw=2,label=labels[i])
            plt.plot(np.arange(full_train_start,len(y_full_wgan)+full_train_start),y_full_wgan,c=colors[i],lw=2,label=labels[i])


            plt.xlabel('iterations')
            plt.title('learning curve')
            plt.legend(loc='upper right')
            plt.gcf().set_size_inches(5., 5., forward=True)

            #plt.show()
            if i == 0: key = 'paper.gan.fulltrain.learning.test.ACC.png'
            if i == 1: key = 'paper.gan.fulltrain.learning.test.MAPE.png'
            plt.savefig(root + '/pic/%s'%key)



def paper_full_train_relistic_situation():
    pass

def paper_full_train_pre_train_is_important():
    pass

if __name__ == '__main__' :
    # paper_fix_train_total_xiao()
    # paper_fix_train_non_self_m_hawkes()
    # paper_hawkes_generator_pretrain_convergence()
    paper_full_train_potential_ability()
    plt.show()
        