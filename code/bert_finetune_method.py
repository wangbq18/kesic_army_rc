#! -*- coding:utf-8 -*-

import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs
import sys 
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

maxlen = 160

base_path = "/media/yinshuai/d8644f6c-5a97-4e12-909b-b61d2271b61c/nlp-datasets"
config_path = os.path.join(base_path, 'chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json')
checkpoint_path = os.path.join(base_path, 'chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt')
dict_path = os.path.join(base_path, 'chinese-bert_chinese_wwm_L-12_H-768_A-12/vocab.txt')

# data_path = '/media/yinshuai/d8644f6c-5a97-4e12-909b-b61d2271b61c/nlp-datasets/senti_dataset'



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




data = np.load('kesic_bert.npy', allow_pickle=True).tolist()



# 按照9:1的比例划分训练集和验证集
random_order = [x for x in range(len(data))]
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]




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
    def __init__(self, data, batch_size=16):
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
            X1, X2, ANS_START_POS, ANS_END_POS, PASSAGE_MASK, HAS_ANS = [], [], [], [], [], []
            for i in idxs:
                d = self.data[i]
                x1, x2 = tokenizer.encode(first=d['passage'])
                #_x1, _x2 = tokenizer.encode(first=d['question'])
                #x1.extend(_x1)
                #x2.extend(_x2)


                passage_mask = [0] + [1] * len(d['passage']) + [0]


                ans_start_pos = np.zeros(len(d['passage']) + 2 , dtype='int32')
                ans_end_pos = np.zeros(len(d['passage']) + 2 , dtype='int32')

                has_ans = [1]
                if d['answer'] != '':
                    answer_start_idx = d['passage'].index(d['answer'])
                    ans_start_pos[answer_start_idx + 1] = 1 
                    ans_end_pos[answer_start_idx + len(d['answer'])] = 1
                    has_ans = [2]

                
                X1.append(x1)
                X2.append(x2)
                ANS_START_POS.append(ans_start_pos)
                ANS_END_POS.append(ans_end_pos)
                PASSAGE_MASK.append(passage_mask)
                HAS_ANS.append(has_ans)
                
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    ANS_END_POS = seq_padding(ANS_END_POS, maxlen=X1.shape[1])
                    ANS_START_POS = seq_padding(ANS_START_POS, maxlen=X1.shape[1])
                    PASSAGE_MASK = seq_padding(PASSAGE_MASK, maxlen=X1.shape[1])
                    HAS_ANS = seq_padding(HAS_ANS)
                    yield [X1, X2, ANS_START_POS, ANS_END_POS, PASSAGE_MASK, HAS_ANS], None 
                    X1, X2, ANS_START_POS, ANS_END_POS, PASSAGE_MASK, HAS_ANS = [], [], [], [], [], []




train_D = data_generator(train_data)
valid_D = data_generator(valid_data)





bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,), dtype='int32')
x2_in = Input(shape=(None,))
ans_start_pos_in = Input(shape=(None,), dtype='int32' )
ans_end_pos_in = Input(shape=(None,),  dtype='int32')
passage_mask_in = Input(shape=(None,))
has_ans_in = Input(shape=(1,), dtype='int32')


x = bert_model([x1_in, x2_in])

x_cls = Lambda(lambda x: x[:, 0])(x)

p_has_ans = Dense(1, activation='sigmoid')(x_cls)  # 确定该文档是否含有答案


x = Dense(units=128, activation='relu')(x) 
x = Dropout(0.1)(x)
ans_start = Dense(units=2, activation='softmax')(x)
ans_end = Dense(units=2, activation='softmax')(x)

passage_mask = passage_mask_in


p_has_ans_loss = K.sparse_categorical_crossentropy(has_ans_in, p_has_ans)
p_has_ans_loss = K.mean(p_has_ans_loss)


p_ans_start_loss = K.sparse_categorical_crossentropy(ans_start_pos_in, ans_start)
p_ans_start_loss = K.sum(p_ans_start_loss * passage_mask) / K.sum(passage_mask) 
p_ans_end_loss = K.sparse_categorical_crossentropy(ans_end_pos_in, ans_end)
p_ans_end_loss = K.sum(p_ans_end_loss * passage_mask) / K.sum(passage_mask) 


loss = p_has_ans_loss + p_ans_start_loss + p_ans_end_loss



los#s = p_has_ans_loss


learning_rate = 1e-6


model = Model([x1_in, x2_in, ans_start_pos_in, ans_end_pos_in, passage_mask_in, has_ans_in], [p_has_ans, ans_start, ans_end])
'''
model = Model([x1_in, x2_in, ans_start_pos_in, ans_end_pos_in, passage_mask_in, has_ans_in], p_has_ans)
'''
model.add_loss(loss)
model.compile(optimizer=Adam(learning_rate))
print(model.summary())








model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    callbacks=[]
)
