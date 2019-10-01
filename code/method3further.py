#! -*- coding:utf-8 -*-

import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs
from keras.callbacks import Callback
import tqdm 
maxlen = 512 # 450 is ok 
import sys 
from utils import write_csv


import tensorflow as tf 



base_path = "/media/yinshuai/d8644f6c-5a97-4e12-909b-b61d2271b61c/nlp-datasets"
config_path = os.path.join(base_path, 'chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json')
checkpoint_path = os.path.join(base_path, 'chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt')
dict_path = os.path.join(base_path, 'chinese-bert_chinese_wwm_L-12_H-768_A-12/vocab.txt')

data_path = '/media/yinshuai/d8644f6c-5a97-4e12-909b-b61d2271b61c/nlp-datasets/senti_dataset'
model_path = '/media/yinshuai/d8644f6c-5a97-4e12-909b-b61d2271b61c/nlp-datasets'
weight_save_path = os.path.join(model_path, 'kesicnl2sql_finetune_method3further.weights') # kesicnl2sql_finetune.weight is good 
token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)




mode = 'test'

if mode == 'train':
    datas = np.load('kesic_new.npy', allow_pickle=True).tolist()
    data = [] 
    for q in datas:
        d = datas[q]
        #print(d)
        data.append((d['para'], d['question'], d['answer_mark'], len(d['answer_mark'])))

    # for d in datas:
    #     if  d['answer'] == '':
    #         data.append((d['passage'] + ' ' + d['question'],  d['answer'],  0))
    #     if len(data) > 8 * lendata:
    #         break 
elif mode == 'test': 
    datas = np.load('kesic_test.npy', allow_pickle=True).tolist()
    test_data = [] 
    for d in datas:
        test_data.append((d['passage'], d['question'], d['question_id']))

#data = data[:100]
#print(data)

# 按照9:1的比例划分训练集和验证集
'''
random_order = [x for x in range(len(data))]
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]
'''
# 评估数据集就别是那种打撒的了
if mode == 'train':
    train_data = data[: int(len(data) * 0.85)]
    valid_data = data[int(len(data) * 0.85):]

    random_order = [x for x in range(len(train_data))]
    np.random.shuffle(random_order)
    train_data = [train_data[idx] for idx in random_order]

    print(len(train_data))

    print(len(valid_data))

def seq_padding(X, padding=0, maxlen=None):
    if maxlen is None:
        L = [len(x) for x in X]
        ML = max(L)
    else:
        ML = maxlen
    return np.array([
        np.concatenate([x[:ML], [padding] * (ML - len(x))]) if len(x[:ML]) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=6):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = [id for id in range(len(self.data))]
            np.random.shuffle(idxs)
            X1, X2, H, Y, ANSWER_POS, PASSAGE_MASK = [], [], [],[], [], []
            #X1, X2,Y  = [], [], []
            for i in idxs:
                d = self.data[i]
                passage = d[0]   ## 切割了
                question = d[1]
                # print(text)
                x1, x2 = tokenizer.encode(first=passage)
                passage_mask = [0] + [1] * len(passage) + [0]

                _x1, _x2 = tokenizer.encode(first=question)
                x1.extend(_x1)
                x2.extend(_x2)


                answer_pos = np.zeros(len(passage) + 2 , dtype='int32')
                h = [len(passage) + 2]

                un_good = False 
                for ans_info in d[2]:
                    if ans_info['answer'] not in passage: 
                        un_good = True 
                        print(passage)
                        continue
                    idx = passage.index(ans_info['answer'])              
                    # ans_start_pos[idx + 1] = 1 
                    # ans_end_pos[idx + len(ans_info['answer'])] = 1 
                    answer_pos[idx + 1: idx + 1 + len(ans_info['answer'])] = 1 
                if un_good == True: continue 
                y = d[3]
                X1.append(x1)
                X2.append(x2)
                PASSAGE_MASK.append(passage_mask)
                ANSWER_POS.append(answer_pos)
                H.append(h)
                
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    
                    PASSAGE_MASK = seq_padding(PASSAGE_MASK, maxlen=X1.shape[1])
                    ANSWER_POS = seq_padding(ANSWER_POS, maxlen=X1.shape[1])
                    H = seq_padding(H)
                
                    yield [X1, X2, H, Y, ANSWER_POS, PASSAGE_MASK], None 
                    X1, X2,  H,  Y, ANSWER_POS, PASSAGE_MASK= [], [], [],[], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, n]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, n, s_size]的向量。
    seq_gather[x, h]
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32') # must int 32 
    return K.tf.batch_gather(seq, idxs)



bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
h_in = Input(shape=(None,))
y_in = Input(shape=(None,))


answer_pos_in = Input(shape=(None,),  dtype='int32')
passage_mask_in = Input(shape=(None,))


x = bert_model([x1_in, x2_in])
x = Dropout(0.2)(x)
#x = Dense(units=128, activation='relu')(x) 
# question前面的cls呢
#q_cls_idx = 256



q_cls = Lambda(seq_gather)([x, h_in]) # header [cls] is selected  [batch_size, header_step, hidden_size]

q_cls = Lambda(lambda x: x[:, 0])(q_cls) 

x_cls = Lambda(lambda x: x[:, 0])(x)


cls_info = concatenate([q_cls, x_cls], axis=-1) 




p = Dense(10, activation='softmax')(cls_info)



# x = Dropout(0.1)(x)
cls_info_dense = Dense(768, activation='relu')(cls_info)

cls_info_dense = Lambda(lambda x: K.expand_dims(x, 1))(cls_info_dense) 
x_answer_pos = add([x, cls_info_dense])

answer_pos = Dense(2, activation='softmax')(x_answer_pos)
passage_mask = passage_mask_in


train_model = Model([x1_in, x2_in, h_in, y_in, answer_pos_in,  passage_mask_in], [p, answer_pos])
model = Model([x1_in, x2_in, h_in, passage_mask_in], [p, answer_pos])
#model = Model([x1_in, x2_in, y_in], [p])

loss_p = K.sparse_categorical_crossentropy(y_in, p) 
loss_p = K.mean(loss_p)

p_ans_pos_loss = K.sparse_categorical_crossentropy(answer_pos_in, answer_pos)
p_ans_pos_loss = K.sum(p_ans_pos_loss * passage_mask) / K.sum(passage_mask)


loss = loss_p + p_ans_pos_loss 

#loss = loss_p 

train_model.add_loss(loss)
train_model.compile(
    optimizer=Adam(10e-5), # 用足够小的学习率
    metrics=['accuracy']
)
train_model.summary()

if mode == 'train':
    train_D = data_generator(train_data)

    #for d in train_D:
    #     pass 
    valid_D = data_generator(valid_data)

# for d in train_D:
#     pass 

#for d in train_D:
#     for i in range(len(d[0])):
#         print(d[0][i].shape)
#     print('-' * 10)
#     # print(d[0][4].shape)

# learning_rate = 5e-5
learning_rate = 7e-5# from 15 to 8 
min_learning_rate = 9e-6
#
model.load_weights(weight_save_path)
ch_good_regex = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9]')
def test(test_data):
    model.load_weights(weight_save_path)
    # csv_f = open("testsubmit.csv", "w") 
    # writer = csv.writer(csv_f)
    right = 0 
    # submit = [['question_id', 'answer']]
    submit = {'question_id': [], 'answer': []}
    cnt = 0 
    all_cnt = 0 
    d_cnt = 0 
    for val in test_data:
        passage, question, q_id = val[0], val[1],  val[2]
        submit['question_id'].append(q_id)
        all_cnt += 1 
        # print(question)
        # print(passage)
        x1, x2 = tokenizer.encode(first=passage)
        passage_mask = [0] + [1] * len(passage) + [0]
        _x1, _x2 = tokenizer.encode(first=question)
        x1.extend(_x1)
        x2.extend(_x2)
        h = [len(passage) + 2]

        p_ans_cnt, ans_pos = model.predict([np.array([x1]), np.array([x2]), np.array([h]),  np.array([passage_mask])])
        p_ans_cnt = p_ans_cnt.argmax(1)[0]
        # print(p_ans_cnt)
        ans_pos = ans_pos[0].argmax(1).tolist()

        if p_ans_cnt == 1 and 1 in ans_pos[:len(passage)]: 
            p_ans_start = ans_pos.index(1)
            p_ans_end = len(ans_pos) - ans_pos[::-1].index(1)
            # print(question)
            # print(p_ans_start, p_ans_end, passage[p_ans_start - 1:p_ans_end - 1])
            ans = passage[p_ans_start - 1:p_ans_end - 1]
            ans = ans.replace(',', '，')
            ans = passage[1:10] if len(ans.strip()) == 0 else ans 
            ans = ans[:30] if len(ans) > 80 else ans 
            ch_find = ch_good_regex.findall(ans)
            if len(ch_find) == 0:
                ans = '未知'
                print(ans)
                

            
            submit['answer'].append(ans)
            
            cnt += 1 
        elif p_ans_cnt == 2:
            submit['answer'].append('未知')

            d_cnt += 1 
        else:
            submit['answer'].append('未知') 

        # print(p_ans_cnt.argmax(1), ans_pos[0].argmax(1))
        continue




        text, q_id = val[0] , val[1]
        
        x1, x2 = tokenizer.encode(text)
        p, ans_start, ans_end = model.predict([np.array([x1]), np.array([x2])])
        startpos = ans_start[0].argmax(1).tolist()
        
        endpos = ans_end[0].argmax(1).tolist()
        submit['question_id'].append(q_id)
        #print('*' * 10)
        if 1 in startpos and 1 in endpos:
            #print(text)

            #print(q_id, p[0],  startpos.index(1), endpos.index(1), text[startpos.index(1) - 1  : endpos.index(1)])
            ans =  text[startpos.index(1) - 1  : endpos.index(1)]
            ans = str(ans.replace(',', ' '))
            # submit.append([q_id, ans.replace(',', ' ')])

            submit['answer'].append(ans)

        elif 1 in startpos:
            ans_may = text[startpos.index(1): startpos.index(1) + 12]
            ans_may = str(ans_may.replace(',', ' '))

            submit['answer'].append(ans_may)
            
        elif 1 in endpos:
            ans_may =  text[endpos.index(1) - 12: endpos.index(1)]
            ans_may = str(ans_may.replace(',', ' '))
            
            submit['answer'].append(ans_may)


        else: 
            print(q_id)
            print(text)
            # submit.append([q_id, ''])
            submit['answer'].append('未知')
    print(cnt, all_cnt, d_cnt)
    # writer.writerows(submit)
    # print(submit)
    write_csv(submit, './testsubmit.csv')
    



#test(test_data)
#import sys 
#sys.exit(0)

if mode == 'test':
    test(test_data)
    import sys 
    sys.exit(0)

def evaluate(valid_data):
    valid_len = len(valid_data)
    model.load_weights(weight_save_path)
    right = 0 
    right_start_may = 0 
    right_end_may = 0 
    for val in valid_data:
        #print(val)
        passage, question, answer = val[0], val[1],  val[2]
        x1, x2 = tokenizer.encode(first=passage)
        passage_mask = [0] + [1] * len(passage) + [0]
        _x1, _x2 = tokenizer.encode(first=question)
        x1.extend(_x1)
        x2.extend(_x2)
        h = [len(passage) + 2]

        p_ans_cnt, ans_pos = model.predict([np.array([x1]), np.array([x2]), np.array([h]),  np.array([passage_mask])])

        ans_pos = ans_pos[0].argmax(1).tolist()
        if  1 in ans_pos[:len(passage)]: 

            right  += 1

        #print(p_ans_cnt.argmax(1), ans_pos[0].argmax(1))
        continue



        ans_pos = ans_pos[0].argmax(1).tolist()
        print(text)
        print(p_ans_cnt)
        print(ans_pos)
        continue 
        #print('*' * 10)
        #print(text)
        pp = p_ans_cnt[0].tolist()
        #if startpos.count(1) > 1:
        #    print(startpos)
        #    print(endpos)


        
        if 1 in startpos and 1 in endpos: #
            #print('*' * 10)
            #print(text)
            # print(pp.index(max(pp)),  startpos.index(1), endpos.index(1), text[startpos.index(1) - 1  : endpos.index(1)], '-------', ans )
            # if text[startpos.index(1) - 1  : endpos.index(1)] == ans[0]['answer']:
                # pass 
            right += 1 
        elif 1 in startpos:
            print(pp.index(max(pp)), 'start_has', startpos.index(1), text[startpos.index(1)  :startpos.index(1) + 8], '-------', ans )
            right_start_may += 1 
        elif 1 in endpos:
            print(pp.index(max(pp)), 'end has', endpos.index(1), text[endpos.index(1) - 8  : endpos.index(1)], '-------', ans )
            
            right_end_may += 1 
        else: 
            #print(pp, ans)
            # print('* wrong*' * 3)
            pass 
            #if ans == '' :right += 1
        
    print(right / valid_len)
    return 0 
    return right / valid_len

#if mode == 'train':
evaluate(valid_data)
import sys 
sys.exit(0)

class Evaluate(Callback):
    def __init__(self):
        self.accs = []
        self.best = 0
        self.passed = 0
        self.stage = 0

    def on_batch_begin(self, batch, logs=None):
        """
            第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
    
        acc = self.evaluate()
        self.accs.append(acc)
        
        if acc >= self.best:
            self.best = acc
            train_model.save_weights(weight_save_path)
        print('acc: %.5f, best acc: %.5f\n' % (acc, self.best))
    def evaluate(self):

        return evaluate(valid_data)


evaluator = Evaluate()
train_model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=300,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D),
    callbacks=[evaluator]
)
