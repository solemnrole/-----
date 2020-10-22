#编写测试函数
import pandas as pd
import numpy as np
import re,os
import tensorflow as tf
from collections import defaultdict
words=pd.read_csv('dict.txt',delimiter=' ',encoding='utf-8',header=None)
words=words.set_index(0)[1]#设置词为index，保留词频列即可。
chars=pd.Series(list("".join(words.index))).value_counts()
chars[:]=range(1,len(chars)+1)
char2id=defaultdict(int,chars.to_dict())
char2id['PAD']=0
id2char={v:k for k,v in char2id.items()}
tag2id={'o':0,'s':1,'b':2,'m':3,'e':4}
id2tag={v:k for k,v in tag2id.items()}


class SEG():
    def __init__(self):
        self.max_seq=48
        self.speed=1
        self.trans={'s':{'s':np.log(0.35),'b':np.log(0.65)},
                       'b':{'m':np.log(0.35),'e':np.log(0.65)},
                       'm':{'m':np.log(0.35),'e':np.log(0.65)},
                       'e':{'s':np.log(0.35),'b':np.log(0.65)}}
        self.tags=['s','b','m','e']
        self.model_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)),'model')

    def load_model(self):
        saver=tf.compat.v1.train.import_meta_graph(os.path.join(self.model_dir,'tf-model-100.meta') )
        graph=tf.get_default_graph()
        tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
        self.input_ids=graph.get_tensor_by_name('seq_input:0')
        self.target=graph.get_tensor_by_name('target:0')
        self.probalitity=graph.get_tensor_by_name('prop:0')
        sess=tf.Session()
        saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))  # 加载变量值
        return sess

    def vertibi(self,sen,emis):
        paths={'s':np.log(emis[0][0]),'b':np.log(emis[0][1])}
        for n in range(1,len(sen)):
            path_={}
            for m in self.tags:
                nows={}
                for p in paths:
                    try:
                        self.trans[p[-1]][m]
                    except:
                        continue
                    F=self.trans[p[-1]][m]+emis[n,tag2id[m]-1]
                    nows[p+m]=F
                if nows:
                    idx=list(nows.values()).index(max(list(nows.values())))
                    key=list(nows.keys())[idx]
                    path_[key]=nows[key]
            paths=path_
        best_path=list(paths.keys())[list(paths.values()).index(max(list(paths.values())))]
        res=[]
        for i,s in enumerate(best_path):
            if s=='s' or s=='b':
                res.append(sen[i])
            else:
                res[-1]+=sen[i]
        return res

    def predict(self,sentences,sess):
        #对句子进行切句。
        results=[]
        for sen in re.split('[^a-zA-Z0-9\u4e00-\u9fa5]+',sentences):
            d_sen=[char2id[i] for i in sen]
            inp_ids=[d_sen+[0]*(self.max_seq-len(d_sen))]
            targ=[[0]*self.max_seq]
            prop=sess.run(self.probalitity,
                          feed_dict={
                              self.input_ids:inp_ids,
                              self.target:targ})
            #如果考虑到效率，在这里直接取最大值即可，想要更加精细一些，可采用viterbi进行解码
            #方法1，直接取每个字的top1.
            if self.speed:
                result=[]
                frist_max_prop_index=np.argmax(prop[0],axis=-1)
                for i,k in enumerate(frist_max_prop_index):
                    if k==0:
                        break
                    if k==1 or k==2 or not len(result):
                        result.append(sen[i])
                    else:
                        result[-1]+=sen[i]
                results.extend(result)
            # 方法2，用veterbi算法求解最优路径.现在从模型可以得到句子的发射概率，
            # 转移概率假设我们已经从先验分布中得到了。
            else:
                results.extend(self.vertibi(sen,prop[0][:len(sen),1:]))
        return results

seg=SEG()
sess=seg.load_model()

while True:
    sen=input('请输入待分词的句子：')
    print(seg.predict(sentences=sen,sess=sess))
