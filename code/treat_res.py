
import numpy as np 
from utils import * 
datas = np.load('test_res.npy', allow_pickle=True).tolist()
q_final_res_dict = {}
for q_id in datas.keys():
    q_final_res_dict.setdefault(q_id, '未知')
    q_cnt_dict = datas[q_id]['q_cnt']
    answer_dict = datas[q_id]['answer']
    q_cnt = 1 
    if q_cnt_dict:
        q_cnt_after_sort = sorted(q_cnt_dict.items(), key=lambda info: info[1], reverse=True)
        q_cnt = q_cnt_after_sort[0][0]
    else:
        q_cnt = 1

    # print(q_cnt)
    # q_cnt_after_sort = sorted(q_cnt.items(), key=lambda info: info[1], reverse=True)
    # q_cnt = q_cnt_after_sort[0][0]
    if answer_dict:
        answer_after_sort = sorted(answer_dict.items(), key=lambda info: info[1], reverse=True)
        print(answer_after_sort)
        q_final_res_dict[q_id] = answer_after_sort[0][0]

        
        if len(answer_after_sort) > 1: 
            q_final_res_dict[q_id] +=  answer_after_sort[1][0]
        if len(answer_after_sort) > 2 and  answer_after_sort[2][0]: 
            q_final_res_dict[q_id] += answer_after_sort[2][0]
        

    else:
        q_final_res_dict[q_id] = '未知'

    
import time 
import re 
# 读取测试数据，然后生成答案文件
ch_good_regex = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9]')
test_path = '../data/test_data_r0.csv'
def get_test_dataset_submit(test_path):
    test_data_json = [] 
    test_data = pd.read_csv(test_path)
    submit = {'question_id': [], 'answer': []}
    for d in test_data.iterrows():
        question_id = d[1]['question_id']
        submit['question_id'].append(question_id)
        print(question_id)
        # assert question_id in q_final_res_dict
        answer = q_final_res_dict.get(question_id, '未知')
        ch_find = ch_good_regex.findall(answer)
        if len(ch_find) == 0:
            answer = '未知'
        answer = str(answer.replace(',', ' '))
        submit['answer'].append(answer)
    write_csv(submit, './testsubmit{}.method7.csv'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))





# get_test_dataset(test_path)
get_test_dataset_submit(test_path)



# 


