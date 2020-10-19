'''
字法标注与HMM模型：目的分词
此算法是基本的序列标注任务，采用HMM的三状态进行分词，转移概率和发射概率可通过标注语料获得或者通过词库也可以获得。
代码实现的思路：1、从字典中构建出转移概率、发射概率、初始概率；2、根据第一步的结果采用viterbi算法求解出最优分词序列；
因为是从字典中构建概率矩阵，因此不存在s-s、s-b、e-s、e-b这样的转移状态.
本文只是简单的说明HMM分词这个算法。
参：https://kexue.fm/archives/3922
'''
import re
from collections import defaultdict,Counter
import numpy as np
f = open('C:/Users/shiwl1/Desktop/folders/data/dpcq.txt', 'r',encoding='utf-8')  # 读取文章
s = f.read()  # 读取为一个字符串
s=re.sub('[^A-Za-z\u4e00-\u9fa5]','',s)

#状态转移概率，因为是从字典中构建概率矩阵，因此不存在s-s、s-b、e-s、e-b这样的转移状态，
# 所以人为的赋值确实的状态转移概率，只是说明一个问题。
trans = {i:Counter() for i in 'sbme'}
emis={i:Counter() for i in 'sbme'}
start_state={'s':0,'b':0}
with open('dict.txt',encoding='utf-8') as r:
    for line in r:
        word,wordf,_=line.split()
        if len(word)==1:
            start_state['s']+=int(wordf)
            emis['s'][word[0]] += int(wordf)
            continue

        start_state['b'] += int(wordf)
        emis['b'][word[0]]+=int(wordf)
        emis['e'][word[-1]]+=int(wordf)
        if len(word)==2:
            trans['b']['e']+=int(wordf)
        else:
            tmp = 'b' + 'm' * (len(word) - 2) + 'e'
            tl = []
            for i in range(2):
                tl += re.findall('(..)', tmp[i:])
            for k_s in tl:
                trans[k_s[0]][k_s[1]] += int(wordf)
            for k in word[1:-1]:
                emis['m'][k]+=int(wordf)
#状态初始概率、状态总概率
start_sum=sum(start_state.values())
start_log_prop={k:np.log(v/start_sum)for k,v in start_state.items()}
log_total_state = {i:np.log(sum(emis[i].values())) for i in 'sbme'}
log_total_trans = {k:np.log(sum(v.values())) for k,v in trans.items()}
log_total_trans['s']=1.0
log_total_trans['e']=1.0
trans['s']['s']=0.35
trans['s']['b']=0.65
trans['e']['s']=0.35
trans['e']['b']=0.65

def hmm_cut(obs):
    T=len(obs)
    paths={i:(np.log(emis[i][obs[0]]+1)-log_total_state[i]) for i in 'sb'}
    for t in range(1, T):
        path_={}
        for n in 'sbme':#逐个查找前一个状态与当前字的状态概率和发射概率最大值
            nows={}
            for p,v in paths.items():
                try:
                    trans[p[-1]][n]
                except:
                    continue
                F=v*(np.log(trans[p[-1]][n])-log_total_trans[p[-1]])*(np.log(emis[n][obs[t]]+1)-log_total_state[n])
                nows[p+n]=F
            if nows:
                k = list(nows.values()).index(max(nows.values()))
                path_[list(nows.keys())[k]] = list(nows.values())[k]
        paths = path_
    print(paths)
    tags=list(paths.keys())[list(paths.values()).index(max(paths.values()))]
    words = [obs[0]]
    print(tags)
    for i in range(1, len(obs)):
        if tags[i] in ['b', 's']:
            words.append(obs[i])
        else:
            words[-1] += obs[i]
    print(words)
# print(forward('轻叹了一口气'))
hmm_cut('轻叹了一口气')
hmm_cut('需要防治规划')
