#coding:utf-8
import json,csv
class PaperScreenor(object) :
    def __init__(self):
        pass
    
    def screen(self,f,self_rate=[-1,2],self_lower=1,nonself_lower=5,type='',limit=10**4) :
        data_raw = []
        str_lines = []
        result = []
        n_conference = 0
        n_journal = 0
        n_self = 0
        n_nonself = 0
        n_self_non_self_paper = 0

        _file = file(f,'r')
        for i,row in enumerate(csv.reader(file(f,'r'))):
            if i % 4 == 2:# year, venue, venue type
            	row = [float(row[1]),row[2],row[3]]
            	if row[2] == 'j' : n_journal += 1
            	if row[2] == 'c' : n_conference += 1
            elif i % 4 == 0: # self
            	row = [float(x) for x in row[1:] if float(x) < 10]
            	n_self += len(row)
            elif i % 4 == 1 : # n_nonself
            	row = [float(x) for x in row[1:] if float(x) < 10]
            	n_nonself += len(row)
            elif i % 4 == 3: # feature
            	_row = [float(x) for x in row[1:]]
            	_max = max(_row)
            	_min = min(_row)
            	row = [(x - _min)/float(_max - _min) for x in _row]
            data_raw.append(row)
            str_lines.append(_file.next())
            if i % 4 == 3:
                if len(data_raw[i-2]) > 0 and len(data_raw[i-3]) / float(len(data_raw[i-2]) + len(data_raw[i-3])) > self_rate[0] \
                    and len(data_raw[i-3]) / float(len(data_raw[i-2]) + len(data_raw[i-3])) < self_rate[1] and (type == '' or type == data[i-1][2]) \
                    and len(data_raw[i-3]) > self_lower and len(data_raw[i-2]) > nonself_lower:
                    result.extend(str_lines[i-3:i+1])
                    n_self_non_self_paper += 1
                    if len(result) > 4 * limit : break
        print json.dumps({
        	'conference rate':float(n_conference)/(n_journal + n_conference),
        	'self cite rate':float(n_self)/(n_self + n_nonself),
        	'paper with appropriate self':n_self_non_self_paper,
        })
        return result
    
    def paper2_to_paper3(self):
        # preprocess
        paper_data = '../data/paper3.txt'
        paper_data_raw = '../data/paper2.txt'
        result = self.screen(paper_data_raw)
        with open(paper_data,'w') as fw:
            fw.writelines(result)