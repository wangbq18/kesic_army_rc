#! -*- coding:utf-8 -*-
import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import numpy as np
import pandas as pd
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs
from keras.callbacks import Callback
import tqdm 
# maxlen = 512 # 450 is ok 
import sys 
from utils import write_csv

import tensorflow as tf 
base_path = "/media/yinshuai/d8644f6c-5a97-4e12-909b-b61d2271b61c/nlp-datasets"
config_path = os.path.join(base_path, 'chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_config.json')
checkpoint_path = os.path.join(base_path, 'chinese-bert_chinese_wwm_L-12_H-768_A-12/bert_model.ckpt')
dict_path = os.path.join(base_path, 'chinese-bert_chinese_wwm_L-12_H-768_A-12/vocab.txt')

data_path = '/media/yinshuai/d8644f6c-5a97-4e12-909b-b61d2271b61c/nlp-datasets/senti_dataset'
model_path = '/media/yinshuai/d8644f6c-5a97-4e12-909b-b61d2271b61c/nlp-datasets'
import time 
weight_save_path = os.path.join(model_path, 'kesicnl2sql_finetune_7_0.84.weights2019-10-02 03:35:52') # kesicnl2sql_finetune.weight is good 
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


mode = 'train'

if mode == 'train':
    datas = np.load('new_train_method_380.npy', allow_pickle=True).tolist()
    data = [] 

    for d in datas:
        # print('*' * 10)
        # print(d)
        if d['q_cnt'] > 2: d['q_cnt'] = 1 
        item = (d['para'], d['question'], d['answer_mark'], d['q_cnt'])
        assert len(d['para'])+  len(d['question']) == 380 
        assert d['q_cnt'] <= 2 
        data.append(item)

elif mode == 'test': 
    datas = np.load('kesic_test508.npy', allow_pickle=True).tolist()
    test_data = [] 
    for d in datas:
        test_data.append((d['passage'], d['question'], d['question_id']))



# data = data[:100]
# print(len(data))
# sys.exit(0)
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
    train_data = data[: int(len(data) * 0.9)]
    valid_data = data[int(len(data) * 0.9):]

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
    def __init__(self, data, batch_size=9):
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
            X1, X2, H, Y, ANS_START_POS, ANS_END_POS, PASSAGE_MASK = [], [], [], [], [], [], []
            #X1, X2,Y  = [], [], []
            for i in idxs:
                d = self.data[i]
                passage, question, answer = d[0], d[1], d[2]   ## 切割了
                x1, x2 = tokenizer.encode(first=passage)
                passage_mask = [0] + [1] * len(passage) + [0]

                _x1, _x2 = tokenizer.encode(first=question)
                x1.extend(_x1)
                x2.extend(_x2)
                # print(x1[-20:])
                # if len(x1) > maxlen:
                #     print('-' * 10)
                #     print(len(passage))
                #     print(len(question))
                #     print(question)
                #     print(len(passage) + len(question))
                
                # x1 = x1[:maxlen]
                # # print(x1)
                # x2 = x2[:maxlen]

                ans_start_pos = np.zeros(len(passage) + 2 , dtype='int32')
                ans_end_pos = np.zeros(len(passage) + 2 , dtype='int32')
                h = [len(passage) + 2]

                un_good = False 
                if answer: # 如果答案是有值的话
                    for ans_info in answer:
                        if len(ans_info['answer'].strip()) == 0 or ans_info['answer'] not in passage: 
                            un_good = True 
                            print(passage)
                            print(d[2])
                            raise 
                            continue
                                   
                        ans_start_pos[ans_info['start_pos'] + 1] = 1 
                        ans_end_pos[ans_info['end_pos'] + 1] = 1 

                # print('-' * 10)
                # print(d[3])
                # print(ans_start_pos.tolist().count(1))
                # print(ans_end_pos.tolist().count(1))
                if un_good == True: continue 


                y = d[3]
                X1.append(x1)
                X2.append(x2)
                PASSAGE_MASK.append(passage_mask)
                ANS_START_POS.append(ans_start_pos)
                ANS_END_POS.append(ans_end_pos)
                H.append(h)
                # print('-----')
                assert (1 in ans_start_pos and 1 in ans_end_pos) or (1 not in ans_start_pos and 1 not  in ans_end_pos)
                # print(1 in ans_start_pos)
                # print(1 in ans_end_pos)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    
                    PASSAGE_MASK = seq_padding(PASSAGE_MASK, maxlen=X1.shape[1])
                    ANS_START_POS = seq_padding(ANS_START_POS, maxlen=X1.shape[1])
                    ANS_END_POS = seq_padding(ANS_END_POS, maxlen=X1.shape[1])
                    H = seq_padding(H)
                    yield [X1, X2, H, Y, ANS_START_POS, ANS_END_POS, PASSAGE_MASK], None 
                    X1, X2,  H,  Y, ANS_START_POS, ANS_END_POS, PASSAGE_MASK= [], [], [],[], [], [], []


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


ans_start_pos_in = Input(shape=(None,), dtype='int32' )
ans_end_pos_in = Input(shape=(None,),  dtype='int32')
passage_mask_in = Input(shape=(None,))


x = bert_model([x1_in, x2_in])

# new add fit overfit 

x = Dense(1000)(x)
x = BatchNormalization()(x)
x = Activation(activation="relu")(x)
x = Dropout(0.3)(x)

x = Dense(768)(x)
x = BatchNormalization()(x)
x = Activation(activation="relu")(x)
x = Dropout(0.3)(x)



q_cls = Lambda(seq_gather)([x, h_in]) # header [cls] is selected  [batch_size, header_step, hidden_size]
q_cls = Lambda(lambda x: x[:, 0])(q_cls) 
x_cls = Lambda(lambda x: x[:, 0])(x)
#cls_info = concatenate([q_cls, x_cls], axis=-1) 
p = Dense(3, activation='softmax')(q_cls)






#cls_info_dense = Dense(768, activation='relu')(cls_info)
#cls_info_dense = Lambda(lambda x: K.expand_dims(x, 1))(cls_info_dense) 
x_answer_pos = add([x, q_cls])
ans_start = Dense(2, activation='softmax')(x_answer_pos)
ans_end = Dense(2, activation='softmax')(x_answer_pos)

passage_mask = passage_mask_in



train_model = Model([x1_in, x2_in, h_in, y_in, ans_start_pos_in,ans_end_pos_in,  passage_mask_in], [p, ans_start, ans_end])
model = Model([x1_in, x2_in, h_in, passage_mask_in], [p, ans_start, ans_end])
#model = Model([x1_in, x2_in, y_in], [p])
# train_model = Model([x1_in, x2_in, h_in, y_in, ans_start_pos_in,ans_end_pos_in,  passage_mask_in], [p])

loss_p = K.sparse_categorical_crossentropy(y_in, p) 
loss_p = K.mean(loss_p)


p_ans_start_loss = K.sparse_categorical_crossentropy(ans_start_pos_in, ans_start)
p_ans_start_loss = K.sum(p_ans_start_loss * passage_mask) / K.sum(passage_mask)
p_ans_end_loss = K.sparse_categorical_crossentropy(ans_end_pos_in, ans_end)
p_ans_end_loss = K.sum(p_ans_end_loss * passage_mask) / K.sum(passage_mask) 
loss = loss_p + p_ans_start_loss +  p_ans_end_loss

# loss = loss_p 

train_model.add_loss(loss)
train_model.compile(
    optimizer=Adam(3e-5), # 用足够小的学习率
    metrics=['accuracy']
)
train_model.summary()

if mode == 'train':
    train_D = data_generator(train_data)

    # for d in train_D:
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
learning_rate = 3e-5# from 15 to 8 
min_learning_rate = 8e-6

# model.load_weights(weight_save_path)
ch_good_regex = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9]')
def test(test_data):
    #model.load_weights(weight_save_path)
    # csv_f = open("testsubmit.csv", "w") 
    # writer = csv.writer(csv_f)
    right = 0 
    # submit = [['question_id', 'answer']]
    submit = {'question_id': [], 'answer': []}
    cnt = 0 
    all_cnt = 0 
    s_cnt = 0 
    e_cnt = 0 
    n_cnt = 0 
    
    for val in test_data:
        passage, question, q_id = val[0], val[1],  val[2]

        # submit['question_id'].append(q_id)
        all_cnt += 1 
        # print(question)
        # print(passage)
        x1, x2 = tokenizer.encode(first=passage)
        passage_mask = [0] + [1] * len(passage) + [0]
        _x1, _x2 = tokenizer.encode(first=question)
        x1.extend(_x1)
        x2.extend(_x2)
        h = [len(passage) + 2]

        p, ans_start, ans_end = model.predict([np.array([x1]), np.array([x2]), np.array([h]),  np.array([passage_mask])])
        startpos = ans_start[0].argmax(1).tolist()
        p_ans_cnt = p.argmax(1)[0]
        endpos = ans_end[0].argmax(1).tolist()
        submit['question_id'].append(q_id)
        # print('p ans cnt is {}'.format(p_ans_cnt))
        #print('*' * 10)
        # print(q_id)
        #print(passage)
        #print(question)
        #print('*' * 10)
        if p_ans_cnt == 2:
            pass 
            # print(startpos)

            # print(endpos)
            # print(startpos.count(1), endpos.count(1))
        if 1 in startpos[:len(passage)] and 1 in endpos[:len(passage)]:
            #print(text)

            #print(q_id, p[0],  startpos.index(1), endpos.index(1), text[startpos.index(1) - 1  : endpos.index(1)])
            ans =  passage[startpos.index(1) - 1  : endpos.index(1)]
            
            

            # print(ans)
            
            ans = passage[1:10] if len(ans.strip()) == 0 else ans 
            ans = ans[:60] if len(ans) > 80 else ans 
            ch_find = ch_good_regex.findall(ans)
            all_cnt += 1 
            if len(ch_find) == 0:
                ans = '未知'
                # print(ans)

            # submit.append([q_id, ans.replace(',', ' ')])
            #print(ans)
            ans = str(ans.replace(',', ' '))

            submit['answer'].append(ans)
        elif 1 in startpos[:len(passage)]:
            # print('*' * 10)
            # print(question)
            ans = passage[startpos.index(1) - 1: startpos.index(1) + 15]
            
            ans = passage[1:10] if len(ans.strip()) == 0 else ans 
            ch_find = ch_good_regex.findall(ans)
            ans = str(ans.replace(',', ' '))
            if len(ch_find) == 0:
                ans = '未知'
            #print("ONLY START:{}".format(ans))
            submit['answer'].append(ans)
            s_cnt += 1 
        elif 1 in endpos[:len(passage)]:
            # print('*' * 10)
            # print(question)
            ans = passage[endpos.index(1) - 1 - 15: endpos.index(1)]
            
            ans = passage[1:10] if len(ans.strip()) == 0 else ans 
            ch_find = ch_good_regex.findall(ans)
            ans = str(ans.replace(',', ' '))
            if len(ch_find) == 0:
                ans = '未知'
            #print('ONLY END:{}'.format(ans))
            submit['answer'].append(ans)
        else:
            # print('未知')
            submit['answer'].append('未知')
            n_cnt += 1 




        # print(p_ans_cnt.argmax(1), ans_pos[0].argmax(1))
        # continue
    # print(all_cnt, s_cnt, e_cnt, n_cnt)
    # writer.writerows(submit)
    # print(submit)
    # print(len(submit['question_id']))
    # print(len(submit['answer']))
    import  time  
    write_csv(submit, './testsubmit{}.method6.csv'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
    

#test(test_data)
#import sys 
#sys.exit(0)

if mode == 'test':
    test(test_data)
    import sys 
    sys.exit(0)


# test_datas = np.load('kesic_test508.npy', allow_pickle=True).tolist()
# test_data = [] 
# for d in test_datas:
#     test_data.append((d['passage'], d['question'], d['question_id']))



def evaluate(valid_data):
    valid_len = len(valid_data)
    # return 0 
    model.load_weights(weight_save_path)
    pos_right = 0 
    neg_right = 0 
    right_start_may = 0 
    right_end_may = 0 
    for val in valid_data:
        passage, question, answer = val[0], val[1],  val[2]
        x1, x2 = tokenizer.encode(first=passage)
        passage_mask = [0] + [1] * len(passage) + [0]
        _x1, _x2 = tokenizer.encode(first=question)
        x1.extend(_x1)
        x2.extend(_x2)
        h = [len(passage) + 2]
        p_ans_cnt, start_pos, end_pos = model.predict([np.array([x1]), np.array([x2]), np.array([h]),  np.array([passage_mask])])
        # print(p_ans_cnt.argmax(1), ans_pos[0].argmax(1))
        startpos = start_pos[0].argmax(1).tolist()
        endpos = end_pos[0].argmax(1).tolist()
        pp = p_ans_cnt[0].tolist()
        #and pp.index(max(pp)) == 1
        # print(answer is None)
        if 1 in startpos and 1 in endpos and answer is not None:
            print(pp,  startpos.index(1), endpos.index(1), passage[startpos.index(1) - 1  : endpos.index(1)], '-------', answer )
            # if text[startpos.index(1) - 1  : endpos.index(1)] == ans:
                # right += 1 
            pos_right += 1 
        if  answer is None and 1 not  in startpos and 1 not in endpos:
            neg_right += 1 
    print(pos_right / valid_len)
    print(neg_right / valid_len)
    # test(test_data)
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
            import time 
            train_model.save_weights(weight_save_path + str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        print('acc: %.5f, best acc: %.5f Acc: %s\n ' % (acc, self.best, str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) ))
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
