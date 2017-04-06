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

from point_process.hawkes import MHawkes, MTPP, Single
from point_process.generator import HawkesGenerator
from point_process.gan import HawkesGAN

np.random.seed(137)
os.environ["KERAS_BACKEND"] = "tensorflow"

import os, sys
root = os.path.abspath(os.path.dirname(__file__))

def draw_fix_train_total_xiao(dataset_id):
    will_train_mtpp = False
    will_train_single = False
    will_draw = True
    # preprocess
    dataset_path = root + '/data/' + dataset_id + '.txt'
    
    # training
    mape_acc_data = root + '/data/' + dataset_id + '.hawkes.mape_acc.txt'
    if will_train_mtpp == True :
        predictor = MTPP()
        loaded = predictor.load(dataset_path,cut=10)
        proposed_10_result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
        loaded = predictor.load(dataset_path,cut=15)
        proposed_15_result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
        loaded = predictor.load(dataset_path,cut=20)
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
        loaded = predictor.load(dataset_path)
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
            if i == 0: key = '' + dataset_id + '.fix-train.total.xiao.mape.png'
            if i == 3: key = '' + dataset_id + '.fix-train.total.xiao.acc.png'
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.legend(fontsize=13)
            plt.savefig(root + '/pic/%s'%key)
        
def draw_fix_train_non_self_m_hawkes(dataset_id):
    will_train_mtpp = False
    will_train_hawkes = False
    will_draw = True
    # preprocess
    dataset_path = root + '/data/' + dataset_id + '.txt'
    
    # training
    mape_acc_data = root + '/data/' + dataset_id + '.hawkes.mape_acc.txt'
    if will_train_mtpp == True :
        predictor = MTPP()
        loaded = predictor.load(dataset_path,cut=10)
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
        loaded = predictor.load(dataset_path,cut=10)
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
            if i == 0: key = '' + dataset_id + '.fix-train.non-self.m-hawks.mape.png'
            if i == 1: key = '' + dataset_id + '.fix-train.non-self.m-hawks.acc.png'
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.legend(fontsize=13)
            plt.savefig(root + '/pic/%s'%key)

def draw_hawkes_generator_pretrain_convergence(dataset_id):
    will_train_hawkes_3_30 = False
    will_train_hawkes_3_20 = False
    will_train_hawkes_3_10 = False
    will_train_hawkes_1_10 = False
    will_draw = True
    # preprocess
    dataset_path = root + '/data/' + dataset_id + '.txt'
    
    # training
    log_pre_train = root + '/data/' + dataset_id + '.pretrain.log.3.30.txt'
    log_pre_train_3_20 = root + '/data/' + dataset_id + '.pretrain.log.3.20.txt'
    log_pre_train_3_10 = root + '/data/' + dataset_id + '.pretrain.log.3.10.txt'
    log_pre_train_1_10 = root + '/data/' + dataset_id + '.pretrain.log.1.10.txt'

    if will_train_hawkes_3_30 == True :
        with open(log_pre_train,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            predictor = HawkesGenerator()
            loaded = predictor.load(dataset_path)
            model = predictor.pre_train(*loaded,max_outer_iter=10)
            sys.stdout = old_stdout

    if will_train_hawkes_3_20 == True :
        with open(log_pre_train_3_20,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            predictor = HawkesGenerator()
            loaded = predictor.load(dataset_path)
            model = predictor.pre_train(*loaded,max_outer_iter=20,alpha_iter=3,w_iter=20)
            sys.stdout = old_stdout

    if will_train_hawkes_3_10 == True :
        with open(log_pre_train_3_10,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            predictor = HawkesGenerator()
            loaded = predictor.load(dataset_path)
            model = predictor.pre_train(*loaded,max_outer_iter=30,alpha_iter=3,w_iter=10)
            sys.stdout = old_stdout

    if will_train_hawkes_1_10 == True :
        with open(log_pre_train_1_10,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            predictor = HawkesGenerator()
            loaded = predictor.load(dataset_path)
            model = predictor.pre_train(*loaded,max_outer_iter=40,alpha_iter=1,w_iter=10)
            sys.stdout = old_stdout
    # drawing
    if will_draw == True :
        # plt.figure(figsize=(8,6), dpi=72, facecolor="white")
        # colors = ['red','green','purple']
        keys = [lambda x:x['LL'], lambda x:x['acc'][-1], lambda x:x['mape'][-1], lambda x:x['acc_val'][-1], lambda x:x['mape_val'][-1]]
        colors = {'3:30':'red','3:20':'green','3:10':'blue','1:10':'purple'}
        # keys = [lambda x:x['acc'][-1], lambda x:x['mape'][-1]]
        labels_prefix = ['Loss','ACC','MAPE','ACC on val.', 'MAPE on val.']
        labels_suffix = {
            '3:30':'3:30',
            '3:20':'3:20',
            '3:10':'3:10',# with optimal $\\beta$',
            '1:10':'1:10',
        }

        f_pre_train = open(log_pre_train)
        nodes_pre_train = []
        f_pre_train_3_20 = open(log_pre_train_3_20)
        nodes_pre_train_3_20 = []
        f_pre_train_3_10 = open(log_pre_train_3_10)
        nodes_pre_train_3_10 = []
        f_pre_train_1_10 = open(log_pre_train_1_10)
        nodes_pre_train_1_10 = []

        for i in range(3):
            plt.figure()

            for line in f_pre_train:
                try:
                    node = eval(line)
                    nodes_pre_train.append(node)
                except:
                    print 'error'

            for line in f_pre_train_3_20:
                try:
                    node = eval(line)
                    nodes_pre_train_3_20.append(node)
                except:
                    print 'error'

            for line in f_pre_train_3_10:
                try:
                    node = eval(line)
                    nodes_pre_train_3_10.append(node)
                except:
                    print 'error'

            for line in f_pre_train_1_10:
                try:
                    node = eval(line)
                    nodes_pre_train_1_10.append(node)
                except:
                    print 'error'

            # arrange layout
            y_pre_train = np.array([float(keys[i](node)) for node in nodes_pre_train])
            y_pre_train_3_20 = np.array([float(keys[i](node)) for node in nodes_pre_train_3_20])
            y_pre_train_3_10 = np.array([float(keys[i](node)) for node in nodes_pre_train_3_10])
            y_pre_train_1_10 = np.array([float(keys[i](node)) for node in nodes_pre_train_1_10])

            delta = max(np.max(y_pre_train),np.max(y_pre_train)) - min(np.min(y_pre_train),np.min(y_pre_train))
            delta /= 30.
            x_left_limit = 0
            x_right_limit = 200
            if y_pre_train[0] > y_pre_train[-1]:
                y_lower_limit = min(np.min(y_pre_train),np.min(y_pre_train)) - delta
                y_upper_limit = 0.25 * np.max(y_pre_train) + 0.75 * np.min(y_pre_train)
            else:
                y_lower_limit = 0.75 * np.max(y_pre_train) + 0.25 * np.min(y_pre_train)
                y_upper_limit = max(np.max(y_pre_train),np.max(y_pre_train)) + delta

            plt.ylim(y_lower_limit, y_upper_limit)
            plt.xlim(0,x_right_limit)

            # draw curve
            plt.plot(np.arange(1,len(y_pre_train_1_10)+1),y_pre_train_1_10,c=colors['1:10'],lw=2,
                label=labels_suffix['1:10'])
            plt.plot(np.arange(1,len(y_pre_train_3_10)+1),y_pre_train_3_10,c=colors['3:10'],lw=2,
                label=labels_suffix['3:10'])
            plt.plot(np.arange(1,len(y_pre_train_3_20)+1),y_pre_train_3_20,c=colors['3:20'],lw=2,
                label=labels_suffix['3:20'])
            plt.plot(np.arange(1,len(y_pre_train)+1),y_pre_train,c=colors['3:30'],lw=2,
                label=labels_suffix['3:30'])


            plt.xlabel('iterations')
            plt.title('learning curve for ' + labels_prefix[i])
            plt.legend(loc='upper right')
            plt.gcf().set_size_inches(5., 5., forward=True)

            if i == 0: key = '' + dataset_id + '.gan.pretrain.learning.NLL.png'
            if i == 1: key = '' + dataset_id + '.gan.pretrain.learning.ACC.png'
            if i == 2: key = '' + dataset_id + '.gan.pretrain.learning.MAPE.png'
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.legend(fontsize=13)
            if i == 0: plt.yticks(fontsize=11)
            plt.savefig(root + '/pic/%s'%key)


def draw_full_train_learning_curve_potential_ability(dataset_id):
    will_train_hawkes = False
    will_train_mse = False
    will_train_gan = False
    will_train_wgan = False
    will_draw = True
    # preprocess
    dataset_path = root + '/data/' + dataset_id + '.txt'

    # pre-training
    log_pre_train = root + '/data/' + dataset_id + '.pretrain.log.3.30.txt'

    if will_train_hawkes == True :
        with open(log_pre_train,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            predictor = HawkesGenerator()
            loaded = predictor.load(dataset_path)
            model = predictor.pre_train(*loaded,max_outer_iter=10)
            sys.stdout = old_stdout

    # full-training
    full_train_mse_log = root + '/data/' + dataset_id + '.fulltrain.mse.log.txt'
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
                gan.gen.sequence_weights = json.load(open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json'))
            except:
                loaded = gan.gen.load(dataset_path)
                gan.gen.pre_train(*loaded,max_outer_iter=pretrain_iter)
                with open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight)
            sys.stdout = old_stdout

    full_train_gan_log = root + '/data/' + dataset_id + '.fulltrain.gan.log.txt'
    if will_train_gan == True:
        mse_weight = 0.
        gan_weight = 1.
        train_gan_method = 'gan'
        with open(full_train_gan_log,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json'))
            except:
                loaded = gan.gen.load(dataset_path)
                gan.gen.pre_train(*loaded,max_outer_iter=pretrain_iter)
                with open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path)
            gan.full_train(*loaded,max_fulltrain_iter=400,train_gan_method=train_gan_method,mse_weight=mse_weight,gan_weight=gan_weight)
            sys.stdout = old_stdout

    full_train_wgan_log = root + '/data/' + dataset_id + '.fulltrain.wgan.log.txt'
    if will_train_wgan == True:
        mse_weight = 0.
        gan_weight = 1.
        train_gan_method = 'wgan'
        with open(full_train_wgan_log,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json'))
            except:
                loaded = gan.gen.load(dataset_path)
                gan.gen.pre_train(*loaded,max_outer_iter=pretrain_iter)
                with open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path)
            gan.full_train(*loaded,max_fulltrain_iter=400,train_gan_method=train_gan_method,mse_weight=mse_weight,gan_weight=gan_weight)
            sys.stdout = old_stdout



    # drawing
    if will_draw == True :
        # plt.figure(figsize=(8,6), dpi=72, facecolor="white")
        colors = ['red','green','purple']
        keys = [lambda x:x['acc'][-1], lambda x:x['mape'][-1]]
        labels = ['ACC on test seq.', 'MAPE on test seq.']

        f_pretrain = open(log_pre_train)
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
            x_right_limit = 200
            if y[0] > y[-1]:
                y_lower_limit = min(np.min(y),np.min(y_full_mse)) - delta
                y_upper_limit = 0.25 * np.max(y) + 0.75 * np.min(y)
            else:
                y_lower_limit = 0.75 * np.max(y) + 0.25 * np.min(y)
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
            if i == 0: key = '' + dataset_id + '.gan.fulltrain.learning.test.ACC.png'
            if i == 1: key = '' + dataset_id + '.gan.fulltrain.learning.test.MAPE.png'
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.legend(fontsize=13)
            plt.savefig(root + '/pic/%s'%key)


def draw_full_train_learning_mle_mse_potential_ability(dataset_id):
    will_train_mle_only = False
    will_train_mle_to_mse = False
    will_train_mse_only = False
    will_train_mse_only_noise = False
    will_train_mse_noise_dropout = False
    will_train_mle_mse_aternative = False
    will_draw = True
    # preprocess
    dataset_path = root + '/data/' + dataset_id + '.txt'

    # pre-training
    log_mle_only = root + '/data/' + dataset_id + '.pretrain.log.3.30.txt'

    if will_train_mle_only == True :
        with open(log_mle_only,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            predictor = HawkesGenerator()
            loaded = predictor.load(dataset_path)
            model = predictor.pre_train(*loaded,max_outer_iter=10)
            sys.stdout = old_stdout

    # full-training
    log_mle_to_mse = root + '/data/' + dataset_id + '.fulltrain.mse.log.txt'
    pretrain_iter = 1
    alpha_iter=3
    w_iter=30
    full_train_start = pretrain_iter * (alpha_iter + w_iter)

    if will_train_mle_to_mse == True :
        mse_weight = 1.
        gan_weight = 0.
        with open(log_mle_to_mse,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json'))
            except:
                loaded = gan.gen.load(dataset_path)
                gan.gen.pre_train(*loaded,max_outer_iter=pretrain_iter)
                with open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight)
            sys.stdout = old_stdout

    log_mse_only = root + '/data/' + dataset_id + '.fulltrain.mse_only.log.txt'
    if will_train_mse_only == True:
        mse_weight = 1.
        gan_weight = 0.
        with open(log_mse_only,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            gan.gen.sequence_weights = None
            # exit()
            loaded = gan.load(dataset_path)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_pretrain=False)
            sys.stdout = old_stdout

    log_mse_only_noise = root + '/data/' + dataset_id + '.fulltrain.mse_only_noise.log.txt'
    if will_train_mse_only_noise == True:
        mse_weight = 1.
        gan_weight = 0.
        with open(log_mse_only_noise,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            gan.gen.sequence_weights = None
            # exit()
            loaded = gan.load(dataset_path)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_pretrain=False,need_noise_dropout=True)
            sys.stdout = old_stdout

    log_mse_noise = root + '/data/' + dataset_id + '.fulltrain.mse_noise.log.txt'
    if will_train_mse_noise_dropout == True:
        mse_weight = 1.
        gan_weight = 0.
        with open(log_mse_noise,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json'))
            except:
                loaded = gan.gen.load(dataset_path)
                gan.gen.pre_train(*loaded,max_outer_iter=pretrain_iter)
                with open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True)
            sys.stdout = old_stdout


    # drawing
    if will_draw == True :
        # plt.figure(figsize=(8,6), dpi=72, facecolor="white")
        colors = {'mle_only':'red','mle_mse':'green','mse_only':'blue','mse_noise':'purple'}
        keys = [lambda x:x['acc'][-1], lambda x:x['mape'][-1]]
        labels_prefix = ['ACC','MAPE']
        labels_suffix = {
            'mle_only':'MLE only',
            'mle_mse':'MSE without noise',
            'mse_only':'MSE only',# with optimal $\\beta$',
            'mse_noise':'MSE after MLE',
        }

        f_mle_only = open(log_mle_only)
        f_mle_to_mse = open(log_mle_to_mse)
        f_mse_only = open(log_mse_only)
        f_mse_only_noise = open(log_mse_only_noise)
        f_mse_noise = open(log_mse_noise)
        nodes_mle_only = []
        nodes_mle_to_mse = []
        nodes_mse_only = []
        nodes_mse_only_noise = []
        nodes_mse_noise = []

        for i in range(len(keys)):
            plt.figure()

            for line in f_mle_only:
                try:
                    node = eval(line)
                    nodes_mle_only.append(node)
                except:
                    print 'error'

            for line in f_mle_to_mse:
                try:
                    node = eval(line)
                    nodes_mle_to_mse.append(node)
                except:
                    print 'error'

            for line in f_mse_only:
                try:
                    node = eval(line)
                    nodes_mse_only.append(node)
                except:
                    print 'error'

            for line in f_mse_only_noise:
                try:
                    node = eval(line)
                    nodes_mse_only_noise.append(node)
                except:
                    print 'error'

            for line in f_mse_noise:
                try:
                    node = eval(line)
                    nodes_mse_noise.append(node)
                except:
                    print 'error'

            # arrange layout
            y_mle_only = np.array([float(keys[i](node)) for node in nodes_mle_only])
            y_mle_to_mse = np.array([float(keys[i](node)) for node in nodes_mle_to_mse])
            y_mse_only = np.array([float(keys[i](node)) for node in nodes_mse_only])
            y_mse_only_noise = np.array([float(keys[i](node)) for node in nodes_mse_only_noise])
            y_mse_noise = np.array([float(keys[i](node)) for node in nodes_mse_noise])

            delta = max(np.max(y_mle_only),np.max(y_mle_to_mse)) - min(np.min(y_mle_only),np.min(y_mle_to_mse))
            delta /= 30.
            x_left_limit = 0
            x_right_limit = 300
            if y_mle_only[0] > y_mle_only[-1]:
                y_lower_limit = min(np.min(y_mle_only),np.min(y_mse_noise)) - delta
                y_upper_limit = 0.25 * np.max(y_mle_only) + 0.75 * np.min(y_mle_only)
            else:
                y_lower_limit = 0.75 * np.max(y_mle_only) + 0.25 * np.min(y_mle_only)
                y_upper_limit = max(np.max(y_mle_only),np.max(y_mse_noise)) + delta

            plt.ylim(y_lower_limit, y_upper_limit)
            plt.xlim(0,x_right_limit)

            # draw curve
            plt.plot(np.arange(1,len(y_mle_only)+1),y_mle_only,c=colors['mle_only'],lw=2,
                label=labels_suffix['mle_only'])
            plt.plot(np.arange(0,len(y_mse_only)+0),y_mse_only,c=colors['mse_only'],lw=2,
                label=labels_suffix['mse_only'])
            plt.plot(np.arange(full_train_start,len(y_mse_noise)+full_train_start),y_mse_noise,c=colors['mse_noise'],lw=2,
                label=labels_suffix['mse_noise'])
            plt.plot(np.arange(full_train_start,len(y_mle_to_mse)+full_train_start),y_mle_to_mse,c=colors['mle_mse'],lw=2,
                label=labels_suffix['mle_mse'])


            plt.xlabel('iterations')
            plt.title('learning curve for ' + labels_prefix[i])
            plt.legend(loc='upper right')
            plt.gcf().set_size_inches(5., 5., forward=True)

            #plt.show()
            if i == 0: key = '' + dataset_id + '.gan.fulltrain.learning.mle_mse.test.ACC.png'
            if i == 1: key = '' + dataset_id + '.gan.fulltrain.learning.mle_mse.test.MAPE.png'
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.legend(fontsize=13)
            plt.savefig(root + '/pic/%s'%key)

def draw_full_train_mape_acc_contrast_mle_mse_potential_ability(dataset_id):
    will_train_mle_only = False
    will_train_mle_to_mse = False
    will_train_mse_only = False
    will_train_mse_only_noise = False
    will_train_mse_noise_dropout = False
    will_train_mle_mse_aternative = False
    will_draw = True
    # preprocess
    dataset_path = root + '/data/' + dataset_id + '.txt'

    # pre-training
    log_mle_only = root + '/data/' + dataset_id + '.pretrain.log.3.30.txt'

    if will_train_mle_only == True :
        with open(log_mle_only,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            predictor = HawkesGenerator()
            loaded = predictor.load(dataset_path)
            model = predictor.pre_train(*loaded,max_outer_iter=10)
            sys.stdout = old_stdout

    # full-training
    log_mle_to_mse = root + '/data/' + dataset_id + '.fulltrain.mse.log.txt'
    pretrain_iter = 1
    alpha_iter=3
    w_iter=30
    full_train_start = pretrain_iter * (alpha_iter + w_iter)

    if will_train_mle_to_mse == True :
        mse_weight = 1.
        gan_weight = 0.
        with open(log_mle_to_mse,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json'))
            except:
                loaded = gan.gen.load(dataset_path)
                gan.gen.pre_train(*loaded,max_outer_iter=pretrain_iter)
                with open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight)
            sys.stdout = old_stdout

    log_mse_only = root + '/data/' + dataset_id + '.fulltrain.mse_only.log.txt'
    if will_train_mse_only == True:
        mse_weight = 1.
        gan_weight = 0.
        with open(log_mse_only,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            gan.gen.sequence_weights = None
            # exit()
            loaded = gan.load(dataset_path)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_pretrain=False)
            sys.stdout = old_stdout

    log_mse_only_noise = root + '/data/' + dataset_id + '.fulltrain.mse_only_noise.log.txt'
    if will_train_mse_only_noise == True:
        mse_weight = 1.
        gan_weight = 0.
        with open(log_mse_only_noise,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            gan.gen.sequence_weights = None
            # exit()
            loaded = gan.load(dataset_path)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_pretrain=False,need_noise_dropout=True)
            sys.stdout = old_stdout

    log_mse_noise = root + '/data/' + dataset_id + '.fulltrain.mse_noise.log.txt'
    if will_train_mse_noise_dropout == True:
        mse_weight = 1.
        gan_weight = 0.
        with open(log_mse_noise,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json'))
            except:
                loaded = gan.gen.load(dataset_path)
                gan.gen.pre_train(*loaded,max_outer_iter=pretrain_iter)
                with open(root + '/data/' + dataset_id + '.pretrain.sequence_weights.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True)
            sys.stdout = old_stdout


    # drawing
    if will_draw == True :
        # plt.figure(figsize=(8,6), dpi=72, facecolor="white")
        colors = {'mle_only':'red','mle_mse':'green','mse_only':'blue','mse_noise':'purple'}
        keys = ['acc','mape']
        labels_prefix = ['ACC','MAPE']
        labels_suffix = {
            'mle_only':'MLE only',
            'mle_mse':'MSE without noise',
            'mse_only':'MSE only',# with optimal $\\beta$',
            'mse_noise':'MSE after MLE',
        }

        f_mle_only = open(log_mle_only)
        f_mle_to_mse = open(log_mle_to_mse)
        f_mse_only = open(log_mse_only)
        f_mse_only_noise = open(log_mse_only_noise)
        f_mse_noise = open(log_mse_noise)
        nodes_mle_only = []
        nodes_mle_to_mse = []
        nodes_mse_only = []
        nodes_mse_only_noise = []
        nodes_mse_noise = []

        for i in range(len(keys)):
            plt.figure()

            for line in f_mle_only:
                try:
                    node = eval(line)
                    nodes_mle_only.append(node)
                except:
                    print 'error'

            for line in f_mle_to_mse:
                try:
                    node = eval(line)
                    nodes_mle_to_mse.append(node)
                except:
                    print 'error'

            for line in f_mse_only:
                try:
                    node = eval(line)
                    nodes_mse_only.append(node)
                except:
                    print 'error'

            for line in f_mse_only_noise:
                try:
                    node = eval(line)
                    nodes_mse_only_noise.append(node)
                except:
                    print 'error'

            for line in f_mse_noise:
                try:
                    node = eval(line)
                    nodes_mse_noise.append(node)
                except:
                    print 'error'

            # arrange layout
            epoch_limit = 300
            y_mle_only = np.array(nodes_mle_only[epoch_limit][keys[i]])
            y_mle_to_mse = np.array(nodes_mle_to_mse[epoch_limit - full_train_start][keys[i]])
            y_mse_only = np.array(nodes_mse_only[epoch_limit][keys[i]])
            y_mse_only_noise = np.array(nodes_mse_only_noise[epoch_limit][keys[i]])
            y_mse_noise = np.array(nodes_mse_noise[epoch_limit - full_train_start][keys[i]])


            # draw curve
            plt.plot(np.arange(0,len(y_mle_only)+0),y_mle_only,c=colors['mle_only'],lw=2,
                label=labels_suffix['mle_only'])
            plt.scatter(np.arange(0,len(y_mle_only)+0),y_mle_only,c=colors['mle_only'],lw=0) 
            plt.plot(np.arange(0,len(y_mse_only)+0),y_mse_only,c=colors['mse_only'],lw=2,
                label=labels_suffix['mse_only'])
            plt.scatter(np.arange(0,len(y_mse_only)+0),y_mse_only,c=colors['mse_only'],lw=0)
            plt.plot(np.arange(0,len(y_mse_noise)+0),y_mse_noise,c=colors['mse_noise'],lw=2,
                label=labels_suffix['mse_noise'])
            plt.scatter(np.arange(0,len(y_mse_noise)+0),y_mse_noise,c=colors['mse_noise'],lw=0)
            plt.plot(np.arange(0,len(y_mle_to_mse)+0),y_mle_to_mse,c=colors['mle_mse'],lw=2,
                label=labels_suffix['mle_mse'])
            plt.scatter(np.arange(0,len(y_mle_to_mse)+0),y_mle_to_mse,c=colors['mle_mse'],lw=0)


            plt.xlabel('years')
            plt.title('metrics by ' + labels_prefix[i])
            plt.legend(loc='upper right')
            plt.gcf().set_size_inches(5., 5., forward=True)

            #plt.show()
            if i == 0: key = '' + dataset_id + '.gan.fulltrain.mape_acc.mle_mse.test.ACC.png'
            if i == 1: key = '' + dataset_id + '.gan.fulltrain.mape_acc.mle_mse.test.MAPE.png'
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.legend(fontsize=13)
            plt.savefig(root + '/pic/%s'%key)

            
def draw_full_train_with_early_stopping(dataset_id):
    pass


if __name__ == '__main__' :
    for dataset_id in ['paper3']:
        draw_fix_train_total_xiao(dataset_id)
        draw_fix_train_non_self_m_hawkes(dataset_id)
        draw_hawkes_generator_pretrain_convergence(dataset_id)
        draw_full_train_learning_curve_potential_ability(dataset_id)
        draw_full_train_learning_mle_mse_potential_ability(dataset_id)
        draw_full_train_mape_acc_contrast_mle_mse_potential_ability(dataset_id)
    plt.show()
