'''
SNS数据挖掘
基于词频、凝固度（信息熵）、自由度（边界熵）抽取词语，可用于词库的构建、新词发现、热词生成等任务；
参考链接：http://www.matrix67.com/blog/archives/5044
         https://kexue.fm/archives/3491
本算法为无监督算法，基于大预料自动抽取词语.
本文采用斗破苍穹小说，大小15.8M。
'''
import re
import pandas as pd
import numpy as np
f = open('C:/Users/shiwl1/Desktop/folders/data/dpcq.txt', 'r',encoding='utf-8')  # 读取文章
s = f.read()  # 读取为一个字符串
s=re.sub('[^0-9-A-Za-z\u4e00-\u9fa5]','',s)

myre={2:'(..)',3:'(...)',4:'(....)'}
max_sep = 4  # 候选词语的最大字数
min_count=10
min_support=30 #凝固度阈值
min_free_su=3 #自由度阈值

ngrams=[]
ngrams.append(pd.Series(list(s)).value_counts())#统计单字的频数，保存的都是凝固度高的词
tsum=ngrams[0].sum()#得到单字频数总和,用于计算凝固度

word=[]
for w in range(2,max_sep+1):
    print('正在生成%s字词……'%w)
    tmp=[]
    for i in range(w):
        tmp+=re.findall(myre[w],s[i:])
    #统计w词频数
    tmp=pd.Series(tmp).value_counts()
    #保留所有的w字词，主要是因为在计算凝固度时会有key_error.
    # 如‘各项目’中的‘各项’如果被过滤掉，在计算‘各项目’的凝固度时会有keyerror出现。
    ngrams.append(tmp)
    #最小频数筛选
    tmp=tmp[tmp>min_count]
    #计算凝固度,词的凝固度最小值都需要大于最小支持度才保留当前词。
    for i in range(w-1):
        qq = list(map(lambda ms: tsum * tmp[ms] / ngrams[len(ms[:i+1])-1][ms[:i+1]]/ngrams[len(ms[i+1:])-1][ms[i+1:]],
               tmp.index))#计算每个词的凝固度。=p(xy)/(p(x)*py(y));={p(xyz)/p(x)/py(yz),p(xyz)/p(xy)/p(z)}
        qq=np.asarray(qq)>min_support
        tmp=tmp[qq]
    word.append(tmp.index)#经过凝固度筛选以后的词

def cal_free(sl):
    sl=-((sl/sl.sum()).apply(np.log)*(sl/sl.sum()))
    return sl.sum()

#自由度筛选
for i in range(2,max_sep+1):
    print('正在进行%s字词自由度筛选(%s)……'%(i,len(word[i-2])))
    # 得到左右邻居
    right_and_left_nibor = []
    # 因为是左右邻居，左右＋2，不理解可以自己举例画一画，这种方式会丢弃第一个词和最后一个词，
    # 因为是全文并成一句话操作的，这个误差就忽略
    for j in range(i+2):
        right_and_left_nibor+=re.findall('(.)%s(.)'%myre[i],s[j:])
    right_and_left_nibor=pd.DataFrame(right_and_left_nibor).set_index(1).sort_index()
    r_index=right_and_left_nibor[~right_and_left_nibor.index.duplicated(keep='first')].index
    index=np.sort(np.intersect1d(word[i-2],r_index))#只取有左右邻居且符合凝固度较高的候选词进行自由度筛选
    nigbor=right_and_left_nibor.loc[index]#选定交集部分的index的候选词以及它的左右邻居
    #左右边界熵进行筛选
    index=index[np.array(list(map(lambda ms:cal_free(pd.Series(nigbor[0][ms]).value_counts()),index)))>min_free_su]
    index=index[np.array(list(map(lambda ms:cal_free(pd.Series(nigbor[2][ms]).value_counts()),index)))>min_free_su]
    word[i-2]=index

##输出词、词频
for i in range(len(word)):
    ngrams[i+1]=ngrams[i+1][word[i]].sort_values(ascending=False)

#保存结果
pd.Series(pd.concat(ngrams[1:])).sort_values(ascending=False).to_csv('_1_new_words.txt',encoding='utf-8',header=False)



