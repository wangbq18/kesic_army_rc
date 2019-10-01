#! -*- coding:utf-8 -*-

import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs


maxlen = 130




base_path = "/media/yinshuai/d8644f6c-5a97-4e12-909b-b61d2271b61c/nlp-datasets"
config_path = os.path.join(base_path, 'chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json')
checkpoint_path = os.path.join(base_path, 'chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt')
dict_path = os.path.join(base_path, 'chinese-bert_chinese_wwm_L-12_H-768_A-12/vocab.txt')

data_path = '/media/yinshuai/d8644f6c-5a97-4e12-909b-b61d2271b61c/nlp-datasets/senti_dataset'


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

datas = np.load('kesic_bert.npy', allow_pickle=True).tolist()
data = [] 
for d in datas:
    if  d['answer'] != '':
        data.append((d['passage'] + ' ' + d['question'], d['answer'], 1))
    else:
        data.append((d['passage'] + ' ' + d['question'],  d['answer'],  0))
        
    



'''
neg = pd.read_excel(os.path.join(data_path, 'neg.xls'), header=None)
pos = pd.read_excel(os.path.join(data_path, 'neg.xls'), header=None)

data = []
print(type(neg))

for d in neg[0]:
    data.append((d, 0))

for d in pos[0]:
    data.append((d, 1))
'''

# 按照9:1的比例划分训练集和验证集
random_order = [x for x in range(len(data))]
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]



valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=48):
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
            #X1, X2,Y, ANS_START_POS, ANS_END_POS,PASSAGE_MASK  = [], [], [],[], [], []
            X1, X2,Y  = [], [], []
            TEST = [] 
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]   ## 切割了
                x1, x2 = tokenizer.encode(first=text)
                passage_mask = [0] + [1] * len(text) + [0]
                ans_start_pos = np.zeros(len(text) + 2 , dtype='int32')
                ans_end_pos = np.zeros(len(text) + 2 , dtype='int32')

                test = np.zeros(len(text) + 2 , dtype='int32')

                if d[1] != '' and d[1] in text:
                    idx = text.index(d[1])
                    ans_start_pos[idx + 1] = 1 
                    ans_end_pos[idx + len(d[1])] = 1 
                else: continue 
                test[1:8] = 1 


                y = d[2]

                X1.append(x1)
                X2.append(x2)
                '''
                ANS_START_POS.append(ans_start_pos)
                ANS_END_POS.append(ans_end_pos)
                PASSAGE_MASK.append(passage_mask)
                '''
                Y.append([y])
                TEST.append(test)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    TEST = seq_padding(TEST)

                    '''
                    ANS_START_POS = seq_padding(ANS_START_POS)
                    ANS_END_POS = seq_padding(ANS_END_POS)
                    PASSAGE_MASK = seq_padding(PASSAGE_MASK)
                    '''


                    #yield [X1, X2, Y, ANS_START_POS, ANS_END_POS, PASSAGE_MASK], None 
                    #X1, X2, Y, ANS_START_POS, ANS_END_POS, PASSAGE_MASK = [], [], [],[], [], []
                    yield [X1, X2, Y, TEST], None 
                    X1, X2, Y = [], [], []
                    TEST = [] 


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam


bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
y_in = Input(shape=(1,))
test_in = Input(shape=(None,))

ans_start_pos_in = Input(shape=(None,), dtype='int32' )
ans_end_pos_in = Input(shape=(None,),  dtype='int32')
passage_mask_in = Input(shape=(None,))


x = bert_model([x1_in, x2_in])
x_cls = Lambda(lambda x: x[:, 0])(x)
p = Dense(1, activation='sigmoid')(x_cls)


x = Dense(units=128, activation='relu')(x) 
x = Dropout(0.1)(x)
p_test = Dense(2, activation='softmax')(x)


'''
x = Dense(units=128, activation='relu')(x) 
x = Dropout(0.1)(x)
ans_start = Dense(1, activation='sigmoid')(x)
ans_end = Dense(1, activation='sigmoid')(x)
passage_mask = passage_mask_in
'''

#model = Model([x1_in, x2_in, y_in,ans_start_pos_in,ans_end_pos_in, passage_mask_in], [p, ans_start, ans_end])
model = Model([x1_in, x2_in, y_in, test_in], [p])

loss_p = K.binary_crossentropy(y_in, p) 
loss_p = K.mean(loss_p)

test_loss = K.sparse_categorical_crossentropy(test_in, p_test)
test_loss = K.mean(test_loss)

'''
p_ans_start_loss = K.sparse_categorical_crossentropy(ans_start_pos_in, ans_start)
p_ans_start_loss = K.sum(p_ans_start_loss * passage_mask) / K.sum(passage_mask)
p_ans_end_loss = K.sparse_categorical_crossentropy(ans_end_pos_in, ans_end)
p_ans_end_loss = K.sum(p_ans_end_loss * passage_mask) / K.sum(passage_mask) 

loss = loss_p + p_ans_start_loss  + p_ans_end_loss 
'''
loss = loss_p + test_loss

model.add_loss(loss)
model.compile(
    optimizer=Adam(1e-6), # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()


train_D = data_generator(train_data)

# for d in train_D:
#     for i in range(len(d[0])):
#         print(d[0][i].shape)
#     print('-' * 10)
#     # print(d[0][4].shape)
valid_D = data_generator(valid_data)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)