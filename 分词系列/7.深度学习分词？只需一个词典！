'''
深度学习分词？只需一个词典！
此算法是基本的序列标注任务，只是不需要有大规模的语料，需要一个词典即可。
很适合拿不到数据的情况，先采用无监督方式得到垂直领域词库，再根据词库抽样生成文本进行训练，得到分词模型。
代码实现的思路：将词典中词频数作为权重，进行词语抽样，形成句子，进行模型的训练。
参：https://kexue.fm/archives/4245
'''
import pandas as pd
import numpy as np
import random
from collections import defaultdict
words=pd.read_csv('dict.txt',delimiter=' ',encoding='utf-8',header=None)
words=words.set_index(0)[1]#设置词为index，保留词频列即可。

chars=pd.Series(list("".join(words.index))).value_counts()
chars[:]=range(1,len(chars)+1)
char2id=defaultdict(int,chars.to_dict())
char2id['PAD']=0
id2char={v:k for k,v in char2id.items()}
tag2id={'o':0,'s':1,'b':2,'m':3,'e':4}

batch_size=1024
max_sep=48
embed_size=128
num_hidden=64
num_classes=5
epoch=1000
max_step=1000
display_step=10
save_step=100
lr=0.1

model_path='model/tf-model'

class Random_Choice:
    def __init__(self, elements, weights):
        d = pd.DataFrame(list(zip(elements, weights)))
        self.elements, self.weights = [], []
        for i,j in d.groupby(1):
            self.weights.append(len(j)*i)
            self.elements.append(tuple(j[0]))
        w=np.cumsum(self.weights).astype(np.float64)#保证词频高的词集合选中的概率会比其他词要高
        self.weights = np.cumsum(self.weights).astype(np.float64)/sum(self.weights)
    def choice(self):
        r = np.random.random()
        w = self.elements[np.where(self.weights >= r)[0][0]]
        return w[np.random.randint(0, len(w))]
def word2tag(s):
    if len(s)==1:
        return 's'
    return 'b'+'m'*(len(s)-2)+'e'

def data_generator():
    wc = Random_Choice(words.index, words)
    x, y = [], []
    while True:
        n = np.random.randint(1, 17)
        seq = [wc.choice() for i in range(n)]
        tag = ''.join([word2tag(i) for i in seq])
        seq = [char2id[i] for i in ''.join(seq)]
        if len(seq) > max_sep:
            continue
        else:
            seq = seq + [0]*(max_sep-len(seq))
            tag = [tag2id[i] for i in tag]
            tag = tag + [0]*(max_sep-len(tag))
            x.append(seq)
            y.append(tag)
        if len(x) == batch_size:
            yield np.array(x), np.array(y)
            x, y= [], []



import tensorflow as tf
from rnncell import CoupledInputForgetGateLSTMCell
seq_input=tf.placeholder(shape=(None,max_sep),dtype=tf.int32,name='seq_input')
target=tf.placeholder(shape=(None,max_sep),dtype=tf.int32,name='target')

embedding=tf.get_variable(shape=(len(char2id),embed_size),dtype=tf.float32,name='embedding')

seq_embedding=tf.nn.embedding_lookup(embedding,seq_input)


def biLSTM_encoder(model_inputs, lstm_dim, real_lengths):
    '''
    :param model_inputs:
    :param lstm_dim:
    :param real_lengths:
    :return:
    '''
    with tf.variable_scope('bilstm-encoder'):
        lstm_cell = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
                lstm_cell[direction] = CoupledInputForgetGateLSTMCell(
                    lstm_dim,
                    use_peepholes=True,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    state_is_tuple=True)
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell["forward"],
            lstm_cell["backward"],
            model_inputs,
            dtype=tf.float32,
            sequence_length=real_lengths)
    return tf.concat(outputs, axis=2)

used=tf.sign(seq_input)
length=tf.reduce_sum(used,axis=1)
real_lengths=tf.cast(length,tf.int32)
output = biLSTM_encoder(seq_embedding, num_hidden,real_lengths)
# lstm_cell_fw = tf.contrib.rnn.LSTMCell(num_hidden)
# lstm_cell_bw = tf.contrib.rnn.LSTMCell(num_hidden)
# out, state = tf.nn.bidirectional_dynamic_rnn(
#     cell_fw=lstm_cell_fw,
#     cell_bw=lstm_cell_bw,
#     inputs=seq_embedding,
#     sequence_length=real_lengths,
#     dtype=tf.float32)
#
# output = tf.concat(out, 2)

weights=tf.get_variable(shape=[2*num_hidden,num_classes],
                        name='fc_w',
                        initializer=tf.contrib.layers.xavier_initializer())
biaes=tf.get_variable(shape=[num_classes],
                        name='fc_b',
                        initializer=tf.zeros_initializer())

logits=tf.add(tf.matmul(output,weights),biaes)
prop=tf.nn.log_softmax(logits,name='prop')

labels=tf.one_hot(target,depth=5)
loss=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='loss')

mean_loss=tf.reduce_mean(loss,name='mean_loss')
optimizer=tf.train.AdamOptimizer(learning_rate=lr)
train_op=optimizer.minimize(mean_loss)
init = tf.global_variables_initializer()
saver=tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init)
    st=0
    for ep in range(epoch):
        i=0
        for x,y in data_generator():
            if i>max_step:
                break
            _=sess.run(train_op,feed_dict={seq_input:x,target:y})
            i+=1
            st+=1
            if i % display_step == 0 or i == 1:
                loss_ = sess.run(mean_loss, feed_dict={seq_input: x, target: y})
                print("Epoch " + str(ep)+" Step " + str(i) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss_))
            if i%save_step==0:
                print('ori     data:      '+"".join([id2char[i] for i in x[0] if i!=0]))
                print('ori     data label:'+str(''.join([str(i) for i in y[0] if i!=0])))
                propabilitity = sess.run(prop, feed_dict={seq_input: x, target: y})
                max_pro=np.argmax(propabilitity,axis=-1)
                print('predict data label:'+str(''.join([str(i) for i in max_pro[0] if i!=0])))
                saver.save(sess,save_path=model_path+'-'+str(st))




