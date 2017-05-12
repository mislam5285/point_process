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
from point_process.pp_gan import HawkesGAN

from preprocess.screen import Screenor

np.random.seed(137)
os.environ["KERAS_BACKEND"] = "tensorflow"

import os, sys
root = os.path.abspath(os.path.dirname(__file__))

def screen():
    will_screen_paper = False
    will_screen_patent = False
    if will_screen_paper == True:
        screenor = Screenor()
        paper_data = root + '/data/paper3.txt'
        paper_data_raw = root + '/data/paper2.txt'
        result = screenor.screen_paper(paper_data_raw)
        print {'number of paper result':len(result)}
        with open(paper_data,'w') as fw:
            fw.writelines(result)

    if will_screen_patent == True:
        screenor = Screenor()
        patent_data = root + '/data/patent3.txt'
        patent_data_raw = root + '/data/patent2.txt'
        result = screenor.screen_patent(patent_data_raw)
        print {'number of patent result':len(result)}
        with open(patent_data,'w') as fw:
            fw.writelines(result)

def draw_fix_train_total_xiao(dataset_id, nb_type=1):
    will_train_mtpp = False
    will_train_single = False
    will_draw = True
    # preprocess
    dataset_path = root + '/data/' + dataset_id + '.txt'
    
    # training
    mape_acc_data = root + '/data/' + dataset_id + '.hawkes.mape_acc.json'
    if will_train_mtpp == True :
        predictor = MTPP()
        loaded = predictor.load(dataset_path,cut=10,nb_type=nb_type)
        proposed_10_result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
        loaded = predictor.load(dataset_path,cut=15,nb_type=nb_type)
        proposed_15_result = predictor.predict(predictor.train(*loaded,max_iter=0,max_outer_iter=0),*loaded)
        loaded = predictor.load(dataset_path,cut=20,nb_type=nb_type)
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
        loaded = predictor.load(dataset_path,nb_type=nb_type)
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
                    plt.plot(np.arange(1,len(y)+1),y,line_type[k],c=colors[j],lw=1.2,label=titles[k] + years[j])
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
        
def draw_fix_train_non_self_m_hawkes(dataset_id, nb_type=1):
    will_train_mtpp = False
    will_train_hawkes = False
    will_draw = True
    # preprocess
    dataset_path = root + '/data/' + dataset_id + '.txt'
    
    # training
    mape_acc_data = root + '/data/' + dataset_id + '.hawkes.mape_acc.json'
    if will_train_mtpp == True :
        predictor = MTPP()
        loaded = predictor.load(dataset_path,cut=10,nb_type=nb_type)
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
        loaded = predictor.load(dataset_path,cut=10,nb_type=nb_type)
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
                    plt.plot(np.arange(1,len(y)+1),y,line_type[k],c=colors[j],lw=1.2,label=model_name[k] + "(" + titles[j] + ")")
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

def draw_hawkes_generator_pretrain_convergence(dataset_id, nb_type=1):
    # will_train_hawkes_3_40 = False
    # will_train_hawkes_3_30 = False
    # will_train_hawkes_3_20 = False
    # will_train_hawkes_3_10 = False
    # will_train_hawkes_1_10 = False
    # will_train_hawkes_1_5 = False
    # will_train_hawkes_1_1 = False
    # will_train_hawkes_5_1 = False
    will_train_hawkes = {'1:1':False,'1:5':False,'5:1':False,'3:3':False}
    will_draw = True
    will_draw_mle_curve = {'1:1':False,'1:5':False,'5:1':False,'3:3':False}

    # preprocess
    dataset_path = root + '/data/' + dataset_id + '.txt'

    to_ratio = lambda x:x.split(':')
    # training
    log_pre_train = {}
    for key in will_train_hawkes:
        log_pre_train[key] = root + '/data/' + dataset_id + '.pretrain.log.' + to_ratio(key)[0] + '.' + to_ratio(key)[1] + '.txt'
    # log_pre_train_3_40 = root + '/data/' + dataset_id + '.pretrain.log.3.40.txt'
    # log_pre_train_3_20 = root + '/data/' + dataset_id + '.pretrain.log.3.20.txt'
    # log_pre_train_3_10 = root + '/data/' + dataset_id + '.pretrain.log.3.10.txt'
    # log_pre_train_1_10 = root + '/data/' + dataset_id + '.pretrain.log.1.10.txt'
    # log_pre_train_1_5 = root + '/data/' + dataset_id + '.pretrain.log.1.5.txt'
    # log_pre_train_1_1 = root + '/data/' + dataset_id + '.pretrain.log.1.1.txt'
    # log_pre_train_5_1 = root + '/data/' + dataset_id + '.pretrain.log.5.1.txt'

    for key in will_train_hawkes:
        if will_train_hawkes[key] == True :
            with open(log_pre_train[key],'w') as f:
                old_stdout = sys.stdout
                sys.stdout = f
                predictor = HawkesGenerator()
                loaded = predictor.load(dataset_path,nb_type=nb_type)
                model = predictor.pre_train(*loaded,max_outer_iter=500/(int(to_ratio(key)[0])+int(to_ratio(key)[1])),
                    alpha_iter=int(to_ratio(key)[0]),w_iter=int(to_ratio(key)[1]))
                sys.stdout = old_stdout

    # if will_train_hawkes_3_40 == True :
    #     with open(log_pre_train_3_40,'w') as f:
    #         old_stdout = sys.stdout
    #         sys.stdout = f
    #         predictor = HawkesGenerator()
    #         loaded = predictor.load(dataset_path,nb_type=nb_type)
    #         model = predictor.pre_train(*loaded,max_outer_iter=12,alpha_iter=3,w_iter=40)
    #         sys.stdout = old_stdout

    # if will_train_hawkes_3_20 == True :
    #     with open(log_pre_train_3_20,'w') as f:
    #         old_stdout = sys.stdout
    #         sys.stdout = f
    #         predictor = HawkesGenerator()
    #         loaded = predictor.load(dataset_path,nb_type=nb_type)
    #         model = predictor.pre_train(*loaded,max_outer_iter=25,alpha_iter=3,w_iter=20)
    #         sys.stdout = old_stdout

    # if will_train_hawkes_3_10 == True :
    #     with open(log_pre_train_3_10,'w') as f:
    #         old_stdout = sys.stdout
    #         sys.stdout = f
    #         predictor = HawkesGenerator()
    #         loaded = predictor.load(dataset_path,nb_type=nb_type)
    #         model = predictor.pre_train(*loaded,max_outer_iter=40,alpha_iter=3,w_iter=10)
    #         sys.stdout = old_stdout

    # if will_train_hawkes_1_10 == True :
    #     with open(log_pre_train_1_10,'w') as f:
    #         old_stdout = sys.stdout
    #         sys.stdout = f
    #         predictor = HawkesGenerator()
    #         loaded = predictor.load(dataset_path,nb_type=nb_type)
    #         model = predictor.pre_train(*loaded,max_outer_iter=45,alpha_iter=1,w_iter=10)
    #         sys.stdout = old_stdout

    # if will_train_hawkes_1_5 == True :
    #     with open(log_pre_train_1_5,'w') as f:
    #         old_stdout = sys.stdout
    #         sys.stdout = f
    #         predictor = HawkesGenerator()
    #         loaded = predictor.load(dataset_path,nb_type=nb_type)
    #         model = predictor.pre_train(*loaded,max_outer_iter=90,alpha_iter=1,w_iter=5)
    #         sys.stdout = old_stdout

    # if will_train_hawkes_1_1 == True :
    #     with open(log_pre_train_1_1,'w') as f:
    #         old_stdout = sys.stdout
    #         sys.stdout = f
    #         predictor = HawkesGenerator()
    #         loaded = predictor.load(dataset_path,nb_type=nb_type)
    #         model = predictor.pre_train(*loaded,max_outer_iter=450,alpha_iter=1,w_iter=1)
    #         sys.stdout = old_stdout

    # if will_train_hawkes_5_1 == True :
    #     with open(log_pre_train_5_1,'w') as f:
    #         old_stdout = sys.stdout
    #         sys.stdout = f
    #         predictor = HawkesGenerator()
    #         loaded = predictor.load(dataset_path,nb_type=nb_type)
    #         model = predictor.pre_train(*loaded,max_outer_iter=90,alpha_iter=5,w_iter=1)
    #         sys.stdout = old_stdout
    # drawing
    if will_draw == True :
        # plt.figure(figsize=(8,6), dpi=72, facecolor="white")
        # colors = ['red','green','purple']
        keys = [lambda x:x['LL'], lambda x:x['acc'][-1], lambda x:x['mape'][-1]]
        keys_val = [lambda x:x['LL'], lambda x:x['acc_val'][-1], lambda x:x['mape_val'][-1]]
        colors = {'test':'red','val':'blue','early_stop':'green','test_best':'purple'}
        # keys = [lambda x:x['acc'][-1], lambda x:x['mape'][-1]]
        labels_prefix = ['Loss','ACC','MAPE']

        f_pre_train = {}
        nodes_pre_train = {}
        for key in will_draw_mle_curve:
            if will_draw_mle_curve[key] == True:
                f_pre_train[key] = open(log_pre_train[key])
                nodes_pre_train[key] = []
        # f_pre_train_3_40 = open(log_pre_train_3_40)
        # nodes_pre_train_3_40 = []
        # f_pre_train_3_20 = open(log_pre_train_3_20)
        # nodes_pre_train_3_20 = []
        # f_pre_train_3_10 = open(log_pre_train_3_10)
        # nodes_pre_train_3_10 = []
        # f_pre_train_1_10 = open(log_pre_train_1_10)
        # nodes_pre_train_1_10 = []

        # f_pre_train_1_5 = open(log_pre_train_1_5)
        # nodes_pre_train_1_5 = []
        # f_pre_train_1_1 = open(log_pre_train_1_1)
        # nodes_pre_train_1_1 = []
        # f_pre_train_5_1 = open(log_pre_train_5_1)
        # nodes_pre_train_5_1 = []

        for key in f_pre_train:
            for line in f_pre_train[key]:
                try:
                    node = eval(line)
                    nodes_pre_train[key].append(node)
                except:
                    print 'error'

        # for line in f_pre_train_3_40:
        #     try:
        #         node = eval(line)
        #         nodes_pre_train_3_40.append(node)
        #     except:
        #         print 'error'

        # for line in f_pre_train_3_20:
        #     try:
        #         node = eval(line)
        #         nodes_pre_train_3_20.append(node)
        #     except:
        #         print 'error'

        # for line in f_pre_train_3_10:
        #     try:
        #         node = eval(line)
        #         nodes_pre_train_3_10.append(node)
        #     except:
        #         print 'error'

        # for line in f_pre_train_1_10:
        #     try:
        #         node = eval(line)
        #         nodes_pre_train_1_10.append(node)
        #     except:
        #         print 'error'

        # for line in f_pre_train_1_5:
        #     try:
        #         node = eval(line)
        #         nodes_pre_train_1_5.append(node)
        #     except:
        #         print 'error'

        # for line in f_pre_train_1_1:
        #     try:
        #         node = eval(line)
        #         nodes_pre_train_1_1.append(node)
        #     except:
        #         print 'error'

        # for line in f_pre_train_5_1:
        #     try:
        #         node = eval(line)
        #         nodes_pre_train_5_1.append(node)
        #     except:
        #         print 'error'

        for i in range(3): # acc or mape or loss
            x_left_limit = 0
            x_right_limit = 200

            y_pre_train = {}
            for key in f_pre_train:
                y_pre_train[key] = np.array(
                    [float(keys[i](node)) for node in nodes_pre_train[key]])[x_left_limit:x_right_limit+1]
            # y_pre_train_3_40 = np.array([float(keys[i](node)) for node in nodes_pre_train_3_40])[x_left_limit:x_right_limit+1]
            # y_pre_train_3_20 = np.array([float(keys[i](node)) for node in nodes_pre_train_3_20])[x_left_limit:x_right_limit+1]
            # y_pre_train_3_10 = np.array([float(keys[i](node)) for node in nodes_pre_train_3_10])[x_left_limit:x_right_limit+1]
            # y_pre_train_1_10 = np.array([float(keys[i](node)) for node in nodes_pre_train_1_10])[x_left_limit:x_right_limit+1]
            # y_pre_train_1_5 = np.array([float(keys[i](node)) for node in nodes_pre_train_1_5])[x_left_limit:x_right_limit+1]
            # y_pre_train_1_1 = np.array([float(keys[i](node)) for node in nodes_pre_train_1_1])[x_left_limit:x_right_limit+1]
            # y_pre_train_5_1 = np.array([float(keys[i](node)) for node in nodes_pre_train_5_1])[x_left_limit:x_right_limit+1]

            y_pre_train_val = {}
            for key in f_pre_train:
                y_pre_train_val[key] = np.array(
                    [float(keys_val[i](node)) for node in nodes_pre_train[key]])[x_left_limit:x_right_limit+1]
            # y_pre_train_3_40_val = np.array([float(keys_val[i](node)) for node in nodes_pre_train_3_40])[x_left_limit:x_right_limit+1]
            # y_pre_train_3_20_val = np.array([float(keys_val[i](node)) for node in nodes_pre_train_3_20])[x_left_limit:x_right_limit+1]
            # y_pre_train_3_10_val = np.array([float(keys_val[i](node)) for node in nodes_pre_train_3_10])[x_left_limit:x_right_limit+1]
            # y_pre_train_1_10_val = np.array([float(keys_val[i](node)) for node in nodes_pre_train_1_10])[x_left_limit:x_right_limit+1]
            # y_pre_train_1_5_val = np.array([float(keys_val[i](node)) for node in nodes_pre_train_1_5])[x_left_limit:x_right_limit+1]
            # y_pre_train_1_1_val = np.array([float(keys_val[i](node)) for node in nodes_pre_train_1_1])[x_left_limit:x_right_limit+1]
            # y_pre_train_5_1_val = np.array([float(keys_val[i](node)) for node in nodes_pre_train_5_1])[x_left_limit:x_right_limit+1]

            curves = []
            for key in f_pre_train:
                curve = {
                    'rate':key,
                    'y_test':y_pre_train[key],
                    'y_val':y_pre_train_val[key],
                }
                curves.append(curve)
            # curves = [
            #     {
            #         'rate':'3:30',
            #         'y_test':y_pre_train,
            #         'y_val':y_pre_train_val,
            #     },
            #     {
            #         'rate':'3:40',
            #         'y_test':y_pre_train_3_40,
            #         'y_val':y_pre_train_3_40_val,
            #     },
            #     {
            #         'rate':'3:20',
            #         'y_test':y_pre_train_3_20,
            #         'y_val':y_pre_train_3_20_val,
            #     },
            #     {
            #         'rate':'3:10',
            #         'y_test':y_pre_train_3_10,
            #         'y_val':y_pre_train_3_10_val,
            #     },
            #     {
            #         'rate':'1:10',
            #         'y_test':y_pre_train_1_10,
            #         'y_val':y_pre_train_1_10_val,
            #     },
            #     {
            #         'rate':'1:5',
            #         'y_test':y_pre_train_1_5,
            #         'y_val':y_pre_train_1_5_val,
            #     },
            #     {
            #         'rate':'1:1',
            #         'y_test':y_pre_train_1_1,
            #         'y_val':y_pre_train_1_1_val,
            #     },
            #     {
            #         'rate':'5:1',
            #         'y_test':y_pre_train_5_1,
            #         'y_val':y_pre_train_5_1_val,
            #     },
            # ]

            for curve in curves: # each curve
                fig = plt.figure()

                # arrange layout
                delta = max(np.max(curve['y_test']),np.max(curve['y_test'])) - min(np.min(curve['y_test']),np.min(curve['y_test']))
                delta /= 30.
                if curve['y_test'][0] > curve['y_test'][-1]:
                    y_lower_limit = min(np.min(curve['y_test']),np.min(curve['y_test'])) - delta
                    y_upper_limit = 0.25 * np.max(curve['y_test']) + 0.75 * np.min(curve['y_test'])
                else:
                    y_lower_limit = 0.75 * np.max(curve['y_test']) + 0.25 * np.min(curve['y_test'])
                    y_upper_limit = max(np.max(curve['y_test']),np.max(curve['y_test'])) + delta

                # draw curve
                plt.ylim(y_lower_limit, y_upper_limit)
                plt.xlim(x_left_limit,x_right_limit)

                plt.plot(np.arange(1,len(curve['y_test'])+1),curve['y_test'],c=colors['test'],lw=1.2,
                    label=labels_prefix[i] + ' on test.')
                if i == 2:
                    j = np.argmin(curve['y_test'])
                    plt.plot([j,j],[y_lower_limit,curve['y_test'][j]+delta],'--',c=colors['test_best'],lw=1.2,
                        label='best point')

                plt.xticks(fontsize=13)
                plt.yticks(fontsize=13)
                plt.legend(loc='center right',fontsize=13)

                if i > 0:
                    # arrange layout
                    delta = max(np.max(curve['y_val']),np.max(curve['y_val'])) - min(np.min(curve['y_val']),np.min(curve['y_val']))
                    delta /= 30.
                    if curve['y_val'][0] > curve['y_val'][-1]:
                        y_lower_limit = min(np.min(curve['y_val']),np.min(curve['y_val'])) - delta
                        y_upper_limit = 0.25 * np.max(curve['y_val']) + 0.75 * np.min(curve['y_val'])
                    else:
                        y_lower_limit = 0.75 * np.max(curve['y_val']) + 0.25 * np.min(curve['y_val'])
                        y_upper_limit = max(np.max(curve['y_val']),np.max(curve['y_val'])) + delta

                    # draw curve
                    ax = plt.subplot(111).twinx()
                    # ax.ylim(y_lower_limit, y_upper_limit)
                    # ax.xlim(x_left_limit,x_right_limit)
                    ax.set_ylim(y_lower_limit, y_upper_limit)

                    ax.plot(np.arange(1,len(curve['y_val'])+1),curve['y_val'],'--',c=colors['val'],lw=1.2,
                        label=labels_prefix[i] + ' on val.')
                    if i == 2:
                        log_pre_train_early_stop = root + '/data/' + dataset_id + '.pretrain.early_stop.stop_point.json'
                        # if will_compute_early_stop == True:
                        early_stop = -1
                        moving_average = curve['y_val'][0]
                        k = 20.
                        th = 0.001
                        for n in range(1,len(curve['y_val'])):
                            av = curve['y_val'][n] * (1./k) + moving_average * (k - 1.) / k
                            if moving_average - av <= th:
                                early_stop = n
                                break
                            moving_average = av

                        try:
                            with open(log_pre_train_early_stop) as fr :
                                result = json.load(fr)
                                if not result.has_key(dataset_id):
                                    result[dataset_id] = {}
                                if not result[dataset_id].has_key(curve['rate']):
                                    result[dataset_id][curve['rate']] = {}
                                result[dataset_id][curve['rate']]['mape_val'] = {'window':k,'threshold':th,'stop_point':early_stop}
                        except:
                            result = {dataset_id:{curve['rate']:{'mape_val':{'window':k,'threshold':th,'stop_point':early_stop}}}}
                        with open(log_pre_train_early_stop,'w') as fw :
                            json.dump(result,fw)

                        with open(log_pre_train_early_stop) as fr:
                            result = json.load(fr)
                            early_stop = result[dataset_id][curve['rate']]['mape_val']['stop_point']

                        if early_stop > 0:
                            ax.plot([early_stop,early_stop],[y_lower_limit,curve['y_val'][j]+delta],'--',c=colors['early_stop'],lw=1.2,
                                label='signal of early stop')
                    plt.xticks(fontsize=13)
                    plt.yticks(fontsize=13)
                    # plt.legend()
                    plt.legend(loc='upper right',fontsize=13) #bbox_to_anchor=(0.31,0.8)
                    # plt.legend(fontsize=13)
                    # plt.gca().add_artist(legend_test)

                plt.xlabel('iterations')
                plt.title('learning curve for ' + labels_prefix[i] + ' when $N_{em}:N_{grad}$=' + curve['rate'])
                plt.gcf().set_size_inches(5., 5., forward=True)

                if i == 0: key = '' + dataset_id + '.gan.pretrain.learning.NLL.' + to_ratio(key)[0] + '.' + to_ratio(key)[1] +'.png'
                if i == 1: key = '' + dataset_id + '.gan.pretrain.learning.ACC.' + to_ratio(key)[0] + '.' + to_ratio(key)[1] +'.png'
                if i == 2: key = '' + dataset_id + '.gan.pretrain.learning.MAPE.' + to_ratio(key)[0] + '.' + to_ratio(key)[1] +'.png'
                if i == 0: plt.yticks(fontsize=11)
                plt.savefig(root + '/pic/%s'%key)


def draw_full_train_learning_gan_convergence(dataset_id, nb_type=1):
    will_train_mle_only = False
    # will_train_gan_only = False
    # will_train_gan_only_noise = False
    # will_train_gan_noise_dropout = False
    # will_train_mle_gan_aternative = False

    will_train_mle_to_wgan = False
    # will_train_wgan_only = False
    # will_train_wgan_only_noise = False

    will_train_mse_noise_dropout = False
    will_train_wgan_noise_dropout = False
    will_train_wgan_noise_sample = False
    will_train_mse_noise_sample = False
    will_train_mse_with_wgan_noise_sample = False



    will_draw = True
    # preprocess
    dataset_path = root + '/data/' + dataset_id + '.txt'

    # pre-training
    ratio = [3,30]
    log_mle_only = root + '/data/' + dataset_id + '.pretrain.log.' + str(ratio[0]) + '.' + str(ratio[1]) + '.txt'

    if will_train_mle_only == True :
        with open(log_mle_only,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            predictor = HawkesGenerator()
            loaded = predictor.load(dataset_path,nb_type=nb_type)
            model = predictor.pre_train(*loaded,max_outer_iter=10)
            sys.stdout = old_stdout

    # full-training
    log_mle_to_wgan = root + '/data/' + dataset_id + '.fulltrain.wgan.log.txt'

    alpha_iter = ratio[0]
    w_iter = ratio[1]

    log_pre_train_early_stop = root + '/data/' + dataset_id + '.pretrain.early_stop.stop_point.json'
    with open(log_pre_train_early_stop) as fr:
        result = json.load(fr)
        ratio_key = str(ratio[0]) + ':' + str(ratio[1])
        early_stop = result[dataset_id][ratio_key]['mape_val']['stop_point']

    full_train_start = early_stop
    assert full_train_start > 0

    if will_train_mle_to_wgan == True :
        mse_weight = 0.
        gan_weight = 1.
        with open(log_mle_to_wgan,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(
                    open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json'))
            except:
                loaded = gan.gen.load(dataset_path,nb_type=nb_type)
                sys.stdout = open(root + '/log/pretrain.log','w')
                gan.gen.pre_train(*loaded,early_stop=full_train_start)
                sys.stdout = f
                with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path,nb_type=nb_type)
            gan.full_train(*loaded,train_gan_method='wgan',max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight)
            sys.stdout = old_stdout

    # log_gan_only = root + '/data/' + dataset_id + '.fulltrain.gan_only.log.txt'
    # if will_train_gan_only == True:
    #     mse_weight = 0.
    #     gan_weight = 1.
    #     with open(log_gan_only,'w') as f:
    #         old_stdout = sys.stdout
    #         sys.stdout = f
    #         gan = HawkesGAN()
    #         gan.gen.sequence_weights = None
    #         # exit()
    #         loaded = gan.load(dataset_path,nb_type=nb_type)
    #         gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_pretrain=False)
    #         sys.stdout = old_stdout

    # log_gan_only_noise = root + '/data/' + dataset_id + '.fulltrain.gan_only_noise.log.txt'
    # if will_train_gan_only_noise == True:
    #     mse_weight = 0.
    #     gan_weight = 1.
    #     stddev = 1.
    #     with open(log_gan_only_noise,'w') as f:
    #         old_stdout = sys.stdout
    #         sys.stdout = f
    #         gan = HawkesGAN()
    #         gan.gen.sequence_weights = None
    #         # exit()
    #         loaded = gan.load(dataset_path,nb_type=nb_type)
    #         gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_pretrain=False,need_noise_dropout=True,stddev=stddev)
    #         sys.stdout = old_stdout

    # log_gan_noise = root + '/data/' + dataset_id + '.fulltrain.gan_noise.log.txt'
    # if will_train_gan_noise_dropout == True:
    #     mse_weight = 0.
    #     gan_weight = 1.
    #     stddev = 1.
    #     with open(log_gan_noise,'w') as f:
    #         old_stdout = sys.stdout
    #         sys.stdout = f
    #         gan = HawkesGAN()
    #         try:
    #             gan.gen.sequence_weights = json.load(
    #                 open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json'))
    #         except:
    #             loaded = gan.gen.load(dataset_path,nb_type=nb_type)
    #             sys.stdout = open(root + '/log/pretrain.log','w')
    #             gan.gen.pre_train(*loaded,early_stop=full_train_start)
    #             sys.stdout = f
    #             with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json','w') as fw:
    #                 json.dump(gan.gen.sequence_weights,fw)
    #         # exit()
    #         loaded = gan.load(dataset_path,nb_type=nb_type)
    #         gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True,stddev=stddev)
    #         sys.stdout = old_stdout


    log_mse_noise = root + '/data/' + dataset_id + '.fulltrain.mse_noise.log.txt'
    if will_train_mse_noise_dropout == True:
        mse_weight = 1.
        gan_weight = 0.
        stddev = 15.
        with open(log_mse_noise,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(
                    open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json'))
            except:
                loaded = gan.gen.load(dataset_path,nb_type=nb_type)
                sys.stdout = open(root + '/log/pretrain.log','w')
                gan.gen.pre_train(*loaded,early_stop=full_train_start)
                sys.stdout = f
                with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path,nb_type=nb_type)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True,stddev=stddev)
            sys.stdout = old_stdout

    log_wgan_noise = root + '/data/' + dataset_id + '.fulltrain.wgan_noise.log.txt'
    if will_train_wgan_noise_dropout == True:
        mse_weight = 0.
        gan_weight = 1.
        stddev = 15.
        wgan_clip = 2.
        with open(log_wgan_noise,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(
                    open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json'))
            except:
                loaded = gan.gen.load(dataset_path,nb_type=nb_type)
                sys.stdout = open(root + '/log/pretrain.log','w')
                gan.gen.pre_train(*loaded,early_stop=full_train_start)
                sys.stdout = f
                with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path,nb_type=nb_type)
            gan.full_train(*loaded,train_gan_method='wgan',max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True,
                stddev=stddev,wgan_clip=wgan_clip)
            sys.stdout = old_stdout

    log_wgan_noise_sample = root + '/data/' + dataset_id + '.fulltrain.wgan_noise_sample.log.txt'
    if will_train_wgan_noise_sample == True:
        mse_weight = 0.
        gan_weight = 1.
        stddev = 10.
        sample_stddev = 10.
        wgan_clip = 2.
        with open(log_wgan_noise_sample,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(
                    open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json'))
            except:
                loaded = gan.gen.load(dataset_path,nb_type=nb_type)
                sys.stdout = open(root + '/log/pretrain.log','w')
                gan.gen.pre_train(*loaded,early_stop=full_train_start)
                sys.stdout = f
                with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path,nb_type=nb_type)
            gan.full_train(*loaded,train_gan_method='wgan',max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True,
                stddev=stddev,sample_stddev=sample_stddev,wgan_clip=wgan_clip)
            sys.stdout = old_stdout


    log_mse_noise_sample = root + '/data/' + dataset_id + '.fulltrain.mse_noise_sample.log.txt'
    if will_train_mse_noise_sample == True:
        mse_weight = 1.
        gan_weight = 0.
        stddev = 15.
        sample_stddev = 15.
        with open(log_mse_noise_sample,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(
                    open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json'))
            except:
                loaded = gan.gen.load(dataset_path,nb_type=nb_type)
                sys.stdout = open(root + '/log/pretrain.log','w')
                gan.gen.pre_train(*loaded,early_stop=full_train_start)
                sys.stdout = f
                with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path,nb_type=nb_type)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True,
                stddev=stddev,sample_stddev=sample_stddev)
            sys.stdout = old_stdout

    log_mse_with_wgan_noise_sample = root + '/data/' + dataset_id + '.fulltrain.mse_wgan.noise_sample.log.txt'
    if will_train_mse_with_wgan_noise_sample == True:
        mse_weight = 0.5
        gan_weight = 0.5
        stddev = 15.
        sample_stddev = 15.
        wgan_clip = 2.
        with open(log_mse_with_wgan_noise_sample,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(
                    open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json'))
            except:
                loaded = gan.gen.load(dataset_path,nb_type=nb_type)
                sys.stdout = open(root + '/log/pretrain.log','w')
                gan.gen.pre_train(*loaded,early_stop=full_train_start)
                sys.stdout = f
                with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path,nb_type=nb_type)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True,
                stddev=stddev,sample_stddev=sample_stddev,wgan_clip=wgan_clip)
            sys.stdout = old_stdout

    # drawing
    if will_draw == True :
        # plt.figure(figsize=(8,6), dpi=72, facecolor="white")
        colors = {
            'mle_only':'red',
            'mle_gan':'green',
            'gan_only':'blue',
            'wgan_noise':'purple',
            'mse_noise':'orange',
            'gan_only_noise':'yellow',
            'gan_noise':'black',
            'mse_with_wgan_noise_sample':'green',
            'mse_noise_sample':'blue',
        }
        keys = [lambda x:x['acc'][-1], lambda x:x['mape'][-1]]
        labels_prefix = ['ACC','MAPE']
        labels_suffix = {
            'mle_only':'MLE only',
            'mle_gan':'GAN without noise',
            'gan_only':'GAN only',# with optimal $\\beta$',
            'wgan_noise':'WGAN',
            'mse_noise':'MSE',
            'gan_only_noise':'gan only noise',
            'gan_noise':'gan_noise',
            'mse_with_wgan_noise_sample':'ppGAN',
            'mse_noise_sample':'MSE with noise',
        }

        f_mle_only = open(log_mle_only)
        f_mle_to_wgan = open(log_mle_to_wgan)
        # f_gan_only = open(log_gan_only)
        # f_gan_only_noise = open(log_gan_only_noise)
        # f_gan_noise = open(log_gan_noise)
        f_wgan_noise = open(log_wgan_noise)
        f_mse_noise = open(log_mse_noise)

        f_wgan_noise_sample = open(log_wgan_noise_sample)
        f_mse_noise_sample = open(log_mse_noise_sample)
        f_mse_with_wgan_noise_sample = open(log_mse_with_wgan_noise_sample)

        nodes_mle_only = []
        nodes_mle_to_wgan = []
        # nodes_gan_only = []
        # nodes_gan_only_noise = []
        # nodes_gan_noise = []
        nodes_wgan_noise = []
        nodes_mse_noise = []

        nodes_wgan_noise_sample = []
        nodes_mse_noise_sample = []
        nodes_mse_with_wgan_noise_sample = []

        for line in f_mle_only:
            try:
                node = eval(line)
                nodes_mle_only.append(node)
            except:
                print 'error'

        for line in f_mle_to_wgan:
            try:
                node = eval(line)
                nodes_mle_to_wgan.append(node)
            except:
                print 'error'

        # for line in f_gan_only:
        #     try:
        #         node = eval(line)
        #         nodes_gan_only.append(node)
        #     except:
        #         print 'error'

        # for line in f_gan_only_noise:
        #     try:
        #         node = eval(line)
        #         nodes_gan_only_noise.append(node)
        #     except:
        #         print 'error'

        # for line in f_gan_noise:
        #     try:
        #         node = eval(line)
        #         nodes_gan_noise.append(node)
        #     except:
        #         print 'error'

        for line in f_wgan_noise:
            try:
                node = eval(line)
                nodes_wgan_noise.append(node)
            except:
                print 'error'

        for line in f_mse_noise:
            try:
                node = eval(line)
                nodes_mse_noise.append(node)
            except:
                print 'error'


        for line in f_wgan_noise_sample:
            try:
                node = eval(line)
                nodes_wgan_noise_sample.append(node)
            except:
                print 'error'

        for line in f_mse_noise_sample:
            try:
                node = eval(line)
                nodes_mse_noise_sample.append(node)
            except:
                print 'error'

        for line in f_mse_with_wgan_noise_sample:
            try:
                node = eval(line)
                nodes_mse_with_wgan_noise_sample.append(node)
            except:
                print 'error'

        for i in range(len(keys)):
            plt.figure()

            # arrange layout
            y_mle_only = np.array([float(keys[i](node)) for node in nodes_mle_only])
            y_mle_to_wgan = np.array([float(keys[i](node)) for node in nodes_mle_to_wgan])
            # y_gan_only = np.array([float(keys[i](node)) for node in nodes_gan_only])
            # y_gan_only_noise = np.array([float(keys[i](node)) for node in nodes_gan_only_noise])
            # y_gan_noise = np.array([float(keys[i](node)) for node in nodes_gan_noise])
            y_wgan_noise = np.array([float(keys[i](node)) for node in nodes_wgan_noise])
            y_mse_noise = np.array([float(keys[i](node)) for node in nodes_mse_noise])

            y_wgan_noise_sample = np.array([float(keys[i](node)) for node in nodes_wgan_noise_sample])
            y_mse_noise_sample = np.array([float(keys[i](node)) for node in nodes_mse_noise_sample])
            y_mse_with_wgan_noise_sample = np.array([float(keys[i](node)) for node in nodes_mse_with_wgan_noise_sample])


            delta = max(np.max(y_mle_only),np.max(y_wgan_noise)) - min(np.min(y_mle_only),np.min(y_wgan_noise))
            delta /= 30.
            x_left_limit = 0
            x_right_limit = 420
            if y_mle_only[0] > y_mle_only[-1]:
                y_lower_limit = min(np.min(y_mle_only),np.min(y_wgan_noise),np.min(y_mse_noise)) - delta
                y_upper_limit = 0.25 * np.max(y_mle_only) + 0.75 * np.min(y_mle_only)
            else:
                y_lower_limit = 0.75 * np.max(y_mle_only) + 0.25 * np.min(y_mle_only)
                y_upper_limit = max(np.max(y_mle_only),np.max(y_wgan_noise),np.max(y_mse_noise)) + delta

            # y_lower_limit = y_lower_limit if y_lower_limit < 0.5 else 0.6
            # y_upper_limit = y_upper_limit if y_upper_limit > 0.5 else 0.4
            plt.ylim(y_lower_limit, y_upper_limit)
            plt.xlim(0,x_right_limit)

            # draw curve
            plt.plot(np.arange(1,len(y_mle_only)+1),y_mle_only,c=colors['mle_only'],lw=1.2,
                label=labels_suffix['mle_only'])
            # plt.plot(np.arange(0,len(y_gan_only)+0),y_gan_only,c=colors['gan_only'],lw=1.2,
            #     label=labels_suffix['gan_only'])
            # plt.plot(np.arange(0,len(y_gan_only_noise)+0),y_gan_only_noise,c=colors['gan_only_noise'],lw=1.2,
            #     label=labels_suffix['gan_only_noise'])
            # plt.plot(np.arange(full_train_start,len(y_gan_noise)+full_train_start),y_gan_noise,c=colors['gan_noise'],lw=1.2,
            #     label=labels_suffix['gan_noise'])
            plt.plot(np.arange(full_train_start,len(y_wgan_noise)+full_train_start),y_wgan_noise,c=colors['wgan_noise'],lw=1.2,
                label=labels_suffix['wgan_noise'])
            # plt.plot(np.arange(full_train_start,len(y_mle_to_wgan)+full_train_start),y_mle_to_wgan,c=colors['mle_gan'],lw=1.2,
            #     label=labels_suffix['mle_gan'])
            plt.plot(np.arange(full_train_start,len(y_mse_noise)+full_train_start),y_mse_noise,c=colors['mse_noise'],lw=1.2,
                label=labels_suffix['mse_noise'])

            plt.plot(np.arange(full_train_start,len(y_wgan_noise_sample)+full_train_start),y_wgan_noise_sample,c=colors['wgan_noise'],lw=1.2,
                label=labels_suffix['wgan_noise'])
            plt.plot(np.arange(full_train_start,len(y_mse_noise_sample)+full_train_start),y_mse_noise_sample,c=colors['mse_noise_sample'],lw=1.2,
                label=labels_suffix['mse_noise_sample'])
            plt.plot(np.arange(full_train_start,len(y_mse_with_wgan_noise_sample)+full_train_start),y_mse_with_wgan_noise_sample,c=colors['mse_with_wgan_noise_sample'],lw=1.2,
                label=labels_suffix['mse_with_wgan_noise_sample'])


            plt.xlabel('iterations')
            plt.title('learning curve for ' + labels_prefix[i])
            plt.legend(loc='upper right')
            plt.gcf().set_size_inches(5., 5., forward=True)

            #plt.show()
            if i == 0: key = '' + dataset_id + '.gan.fulltrain.learning.mle_gan.test.ACC.png'
            if i == 1: key = '' + dataset_id + '.gan.fulltrain.learning.mle_gan.test.MAPE.png'
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.legend(fontsize=13)
            plt.savefig(root + '/pic/%s'%key)




def draw_full_train_learning_mle_mse_convergence(dataset_id, nb_type=1):
    will_train_mle_only = False
    will_train_mle_to_mse = False

    will_train_mse_only = False
    will_train_mse_only_noise = False
    will_train_mse_noise_dropout = False
    will_train_mle_mae = False

    will_draw = True
    # preprocess
    dataset_path = root + '/data/' + dataset_id + '.txt'

    # pre-training
    ratio = [3,30]
    log_mle_only = root + '/data/' + dataset_id + '.pretrain.log.' + str(ratio[0]) + '.' + str(ratio[1]) + '.txt'

    if will_train_mle_only == True :
        with open(log_mle_only,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            predictor = HawkesGenerator()
            loaded = predictor.load(dataset_path,nb_type=nb_type)
            model = predictor.pre_train(*loaded,max_outer_iter=10)
            sys.stdout = old_stdout

    # full-training
    log_mle_to_mse = root + '/data/' + dataset_id + '.fulltrain.mse.log.txt'

    alpha_iter = ratio[0]
    w_iter = ratio[1]

    log_pre_train_early_stop = root + '/data/' + dataset_id + '.pretrain.early_stop.stop_point.json'
    with open(log_pre_train_early_stop) as fr:
        result = json.load(fr)
        ratio_key = str(ratio[0]) + ':' + str(ratio[1])
        early_stop = result[dataset_id][ratio_key]['mape_val']['stop_point']

    full_train_start = early_stop
    assert full_train_start > 0

    if will_train_mle_to_mse == True :
        mse_weight = 1.
        gan_weight = 0.
        with open(log_mle_to_mse,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(
                    open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json'))
            except:
                loaded = gan.gen.load(dataset_path,nb_type=nb_type)
                sys.stdout = open(root + '/log/pretrain.log','w')
                gan.gen.pre_train(*loaded,early_stop=full_train_start)
                sys.stdout = f
                with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path,nb_type=nb_type)
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
            loaded = gan.load(dataset_path,nb_type=nb_type)
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
            loaded = gan.load(dataset_path,nb_type=nb_type)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_pretrain=False,need_noise_dropout=True)
            sys.stdout = old_stdout

    log_mse_noise = root + '/data/' + dataset_id + '.fulltrain.mse_noise.log.txt'
    # if will_train_mse_noise_dropout == True:
    #     mse_weight = 1.
    #     gan_weight = 0.
    #     with open(log_mse_noise,'w') as f:
    #         old_stdout = sys.stdout
    #         sys.stdout = f
    #         gan = HawkesGAN()
    #         try:
    #             gan.gen.sequence_weights = json.load(
    #                 open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json'))
    #         except:
    #             loaded = gan.gen.load(dataset_path,nb_type=nb_type)
    #             sys.stdout = open(root + '/log/pretrain.log','w')
    #             gan.gen.pre_train(*loaded,early_stop=full_train_start)
    #             sys.stdout = f
    #             with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json','w') as fw:
    #                 json.dump(gan.gen.sequence_weights,fw)
    #         # exit()
    #         loaded = gan.load(dataset_path,nb_type=nb_type)
    #         gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=True)
    #         sys.stdout = old_stdout

    log_mle_to_mae = root + '/data/' + dataset_id + '.fulltrain.mae.log.txt'
    if will_train_mle_mae == True:
        mse_weight = 1.
        gan_weight = 0.
        with open(log_mle_to_mae,'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            gan = HawkesGAN()
            try:
                gan.gen.sequence_weights = json.load(
                    open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json'))
            except:
                loaded = gan.gen.load(dataset_path,nb_type=nb_type)
                sys.stdout = open(root + '/log/pretrain.log','w')
                gan.gen.pre_train(*loaded,early_stop=full_train_start)
                sys.stdout = f
                with open(root + '/data/' + dataset_id + '.pretrain.early_stop.sequence_weights.' + str(ratio[0]) + '.' + str(ratio[1]) + '.json','w') as fw:
                    json.dump(gan.gen.sequence_weights,fw)
            # exit()
            loaded = gan.load(dataset_path,nb_type=nb_type)
            gan.full_train(*loaded,max_fulltrain_iter=400,mse_weight=mse_weight,gan_weight=gan_weight,need_noise_dropout=False,hawkes_output_loss='mae')
            sys.stdout = old_stdout


    # drawing
    if will_draw == True :
        # plt.figure(figsize=(8,6), dpi=72, facecolor="white")
        colors = {'mle_only':'red','mle_mse':'green','mle_mae':'orange','mse_only':'blue','mse_noise':'purple'}
        keys = [lambda x:x['acc'][-1], lambda x:x['mape'][-1]]
        labels_prefix = ['ACC','MAPE']
        labels_suffix = {
            'mle_only':'MLE only',
            'mle_mse':'MSE without noise',
            'mle_mae':'MAE without noise',
            'mse_only':'MSE only',# with optimal $\\beta$',
            'mse_noise':'MSE after MLE',
        }

        f_mle_only = open(log_mle_only)
        f_mle_to_mse = open(log_mle_to_mse)
        f_mle_to_mae = open(log_mle_to_mae)
        f_mse_only = open(log_mse_only)
        f_mse_only_noise = open(log_mse_only_noise)
        f_mse_noise = open(log_mse_noise)
        nodes_mle_only = []
        nodes_mle_to_mse = []
        nodes_mle_to_mae = []
        nodes_mse_only = []
        nodes_mse_only_noise = []
        nodes_mse_noise = []

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

        for line in f_mle_to_mae:
            try:
                node = eval(line)
                nodes_mle_to_mae.append(node)
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

        for i in range(len(keys)):
            plt.figure()

            # arrange layout
            y_mle_only = np.array([float(keys[i](node)) for node in nodes_mle_only])
            y_mle_to_mse = np.array([float(keys[i](node)) for node in nodes_mle_to_mse])
            y_mle_to_mae = np.array([float(keys[i](node)) for node in nodes_mle_to_mae])
            y_mse_only = np.array([float(keys[i](node)) for node in nodes_mse_only])
            y_mse_only_noise = np.array([float(keys[i](node)) for node in nodes_mse_only_noise])
            y_mse_noise = np.array([float(keys[i](node)) for node in nodes_mse_noise])

            delta = max(np.max(y_mle_only),np.max(y_mle_to_mse)) - min(np.min(y_mle_only),np.min(y_mle_to_mse))
            delta /= 30.
            x_left_limit = 0
            x_right_limit = 420
            if y_mle_only[0] > y_mle_only[-1]:
                y_lower_limit = min(np.min(y_mle_only),np.min(y_mse_noise)) - delta
                y_upper_limit = 0.25 * np.max(y_mle_only) + 0.75 * np.min(y_mle_only)
            else:
                y_lower_limit = 0.75 * np.max(y_mle_only) + 0.25 * np.min(y_mle_only)
                y_upper_limit = max(np.max(y_mle_only),np.max(y_mse_noise)) + delta

            plt.ylim(y_lower_limit, y_upper_limit)
            plt.xlim(0,x_right_limit)

            # draw curve
            plt.plot(np.arange(1,len(y_mle_only)+1),y_mle_only,c=colors['mle_only'],lw=1.2,
                label=labels_suffix['mle_only'])
            plt.plot(np.arange(0,len(y_mse_only)+0),y_mse_only,c=colors['mse_only'],lw=1.2,
                label=labels_suffix['mse_only'])
            plt.plot(np.arange(full_train_start,len(y_mse_noise)+full_train_start),y_mse_noise,c=colors['mse_noise'],lw=1.2,
                label=labels_suffix['mse_noise'])
            plt.plot(np.arange(full_train_start,len(y_mle_to_mse)+full_train_start),y_mle_to_mse,c=colors['mle_mse'],lw=1.2,
                label=labels_suffix['mle_mse'])
            plt.plot(np.arange(full_train_start,len(y_mle_to_mae)+full_train_start),y_mle_to_mae,c=colors['mle_mae'],lw=1.2,
                label=labels_suffix['mle_mae'])


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

def draw_full_train_mle_mse_mape_acc_constast(dataset_id, nb_type=1):
    will_draw = True
    # preprocess
    dataset_path = root + '/data/' + dataset_id + '.txt'

    # pre-training
    ratio = [3,30]
    log_mle_only = root + '/data/' + dataset_id + '.pretrain.log.' + str(ratio[0]) + '.' + str(ratio[1]) + '.txt'

    # full-training
    log_mle_to_mse = root + '/data/' + dataset_id + '.fulltrain.mse.log.txt'

    alpha_iter = ratio[0]
    w_iter = ratio[1]

    log_pre_train_early_stop = root + '/data/' + dataset_id + '.pretrain.early_stop.stop_point.json'
    with open(log_pre_train_early_stop) as fr:
        result = json.load(fr)
        ratio_key = str(ratio[0]) + ':' + str(ratio[1])
        early_stop = result[dataset_id][ratio_key]['mape_val']['stop_point']

    full_train_start = early_stop
    assert full_train_start > 0

    log_mse_only = root + '/data/' + dataset_id + '.fulltrain.mse_only.log.txt'

    log_mse_only_noise = root + '/data/' + dataset_id + '.fulltrain.mse_only_noise.log.txt'

    log_mse_noise = root + '/data/' + dataset_id + '.fulltrain.mse_noise.log.txt'


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

        for i in range(len(keys)):
            plt.figure()

            # arrange layout
            epoch_limit = 420
            y_mle_only = np.array(nodes_mle_only[epoch_limit][keys[i]])
            y_mle_to_mse = np.array(nodes_mle_to_mse[epoch_limit - full_train_start][keys[i]])
            y_mse_only = np.array(nodes_mse_only[epoch_limit][keys[i]])
            y_mse_only_noise = np.array(nodes_mse_only_noise[epoch_limit][keys[i]])
            y_mse_noise = np.array(nodes_mse_noise[epoch_limit - full_train_start][keys[i]])


            # draw curve
            plt.plot(np.arange(1,len(y_mle_only)+1),y_mle_only,c=colors['mle_only'],lw=1.2,
                label=labels_suffix['mle_only'])
            plt.scatter(np.arange(1,len(y_mle_only)+1),y_mle_only,c=colors['mle_only'],lw=0) 
            plt.plot(np.arange(1,len(y_mse_only)+1),y_mse_only,c=colors['mse_only'],lw=1.2,
                label=labels_suffix['mse_only'])
            plt.scatter(np.arange(1,len(y_mse_only)+1),y_mse_only,c=colors['mse_only'],lw=0)
            plt.plot(np.arange(1,len(y_mse_noise)+1),y_mse_noise,c=colors['mse_noise'],lw=1.2,
                label=labels_suffix['mse_noise'])
            plt.scatter(np.arange(1,len(y_mse_noise)+1),y_mse_noise,c=colors['mse_noise'],lw=0)
            plt.plot(np.arange(1,len(y_mle_to_mse)+1),y_mle_to_mse,c=colors['mle_mse'],lw=1.2,
                label=labels_suffix['mle_mse'])
            plt.scatter(np.arange(1,len(y_mle_to_mse)+1),y_mle_to_mse,c=colors['mle_mse'],lw=0)


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

            
def draw_full_train_mle_mse_noise_contrast(dataset_id, nb_type=1):
    pass


if __name__ == '__main__' :
    screen()
    event_types = {
        'paper3':1,
        'patent3':2,
        'patent2':2,
    }
    for dataset_id in ['paper3']:
        # draw_fix_train_total_xiao(dataset_id,nb_type=event_types[dataset_id])
        # draw_fix_train_non_self_m_hawkes(dataset_id,nb_type=event_types[dataset_id])
        draw_hawkes_generator_pretrain_convergence(dataset_id,nb_type=event_types[dataset_id])
        # draw_full_train_learning_gan_convergence(dataset_id,nb_type=event_types[dataset_id])
        # draw_full_train_learning_mle_mse_convergence(dataset_id,nb_type=event_types[dataset_id])
        # draw_full_train_mle_mse_mape_acc_constast(dataset_id,nb_type=event_types[dataset_id])
        pass
    plt.show()
