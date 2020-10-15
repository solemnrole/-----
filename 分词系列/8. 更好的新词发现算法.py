'''
基于切分的新词发现
此算法是在sns基础上进行了变种，主要是由于考虑了边界熵以后，效率比较低；
本代码是换了一个思路，原来思路是先求凝固度较大的词，在此基础上再进行二次边界熵求解，
反过来想也就是如果两个字凝固度不高，则应该将其断开，随后再根据词频统计排序，直接获得词库。
代码实现的思路：1、求凝固度较高的2、3、4字词；2、根据第一步的结果对原文进行切分；3、统计词频
参：https://kexue.fm/archives/3913
第一步，统计：选取某个固定的n，统计2grams、3grams、…、ngrams，计算它们的内部凝固度，
       只保留高于某个阈值的片段，构成一个集合G；
       这一步，可以为2grams、3grams、…、ngrams设置不同的阈值，
       不一定要相同，因为字数越大，一般来说统计就越不充分，越有可能偏高，所以字数越大，阈值要越高；
第二步，切分：用上述grams对语料进行切分（粗糙的分词），并统计频率。
       切分的规则是，当前片段的子片段部分或者全部都在集合G中且子片段之间有重合且可以组成片段即保留此片段。
       比如“轻叹了一口气”，从左向右遍历，假如’轻叹‘、’轻叹了‘、‘轻叹了一’、‘一口气’在凝聚度较高的集合中，
       则‘轻叹了一口气’就保留下来；当然可以有不同的组合，如’轻叹‘,‘轻叹了'、’叹了一口气‘，这样的组合使得
       片段也可以保留下来。这步骤主要是为了多切出一些长词出来。
第三步，回溯：经过第二步，“轻叹了一口气”会被切出来（因为第二步保证宁放过，不切错）。
       这步骤是为了说明如果一个词可以成词，那么它的子词也当是凝固度高的；检查只检查片段拥有长度为3以及4的子片段即可
       （当然也可以加入长度为2的子片段，这里只检查3,4，可以看做对成词的要求降低一些，毕竟目的就是为了多一些长词）
       也就是说如果片段长度为>=3字，需要检测片段所有3字以及4字的子片段是否在凝聚度较高的集合G中，只要有一个子片段不在，就丢弃。
      如：‘轻叹了一口气’的3字4字子片段有：’轻叹了‘、’叹了一‘、’了一口‘、’一口气‘、’轻叹了一‘、
      ’叹了一口‘、’了一口气‘；检查上述片段是否都在集合G中，如果全部在，则保留’轻叹了一口气‘作为词库中的词，否则丢弃；
      最后统计词频，按照词频高低进行排序即可。
'''
import re
import numpy as np
import pandas as pd
f = open('C:/Users/shiwl1/Desktop/folders/data/dpcq.txt', 'r',encoding='utf-8')  # 读取文章
s = f.read()  # 读取为一个字符串
s=re.sub('[^A-Za-z\u4e00-\u9fa5]','',s)

min_count=10#词频
min_support=30#词频
max_step=4
min_proba = {2:5, 3:25, 4:125}
myre={2:'(..)',3:'(...)',4:'(....)'}

d=pd.Series(list(s)).value_counts()
d=d[d>min_count]
tsum=d.sum()#用于计算凝固度
ngrams=[]
ngrams.append(d)

high_word=[]
for j in range(2,max_step+1):
    ngrams.append([])
    for i in range(j):
        ngrams[-1]+=re.findall('%s'%myre[j],s[i:])
    ngrams[-1]=pd.Series(ngrams[-1]).value_counts()
    ngrams[-1]=ngrams[-1][ngrams[-1]>min_count]
    tt=ngrams[-1][:]
    for k in range(j-1):
        qq=np.asarray(list(map(lambda ms:
                               tsum * tt[ms]/ngrams[len(ms[:k+1])-1][ms[:k+1]]/
                               ngrams[len(ms[k+1:])-1][ms[k+1:]],
                               tt.index)))>min_proba[j]
        tt=tt[qq]
    high_word.extend(tt.index)

high_word=set(high_word)
#第二步，切分源文章，进行粗略分词。思路是子片段在集合G中,它所经过的路径位置都加1.
rr=np.array([0]*(len(s)-1))
for i in range(len(s)-1):
    for j in range(2,max_step+1):
        if s[i:i+j] in high_word:
            rr[i:i+j-1] += 1
anchor_word = [s[0]]
for i in range(1, len(s)):
    if rr[i - 1] > 0:
        anchor_word[-1] += s[i]
    else:
        anchor_word.append(s[i])

def check(ms):
    if len(ms)==1:
        return False
    if len(ms)<3:
        return True
    child_=[]
    for i in range(3):
        child_+=re.findall('(...)',ms[i:])
    if len(ms)>3:
        for i in range(4):
            child_+=re.findall('(....)',ms[i:])
    for key in child_:
        if key not in high_word:
            return False
    return True

#第三步，检查
anchor_word=pd.Series(anchor_word).value_counts()
index=np.asarray(list(map(lambda ms:check(ms),anchor_word.index)))
last_word=anchor_word[index]

last_word.sort_values(ascending=False).to_csv('3_result.txt',encoding='utf-8',header=False)
