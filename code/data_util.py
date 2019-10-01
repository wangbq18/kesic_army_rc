import os 
import sys 
import pandas  as pd 

test_path = '../data/test_data_r0.csv'
train_path = '../data/train_round_0.csv'


train_data = pd.read_csv(train_path)




print(train_data.columns)

'''
Index(['answer', 'bridging_entity', 'content1', 'content2', 'content3',
       'content4', 'content5', 'keyword', 'question', 'supporting_paragraph',
       'title1', 'title2', 'title3', 'title4', 'title5', 'question_id'],
      dtype='object')
'''
      
print(train_data.head(3))
from collections import Counter
train_data['content1_sen'] = train_data['content1'].str.split('。')
sen_len_stat = [] 
sen_cnt = 0 
for sen_list in train_data['content1_sen']:
    for sen in sen_list:
        sen_cnt += 1
        sen_len_stat.append(len(sen))

print(Counter(sen_len_stat).most_common(200))
print(sen_cnt)
train_data['question_len'] = train_data['question'].str.len()
print(train_data['question_len'].describe())
print("question len of 99% is {}".format(train_data['question_len'].quantile(.99)))


sys.exit(0)
import matplotlib.pyplot as plt
xdim = [] 
ydim = []
for couter in Counter(sen_len_stat).most_common(200):
    xdim.append(couter[0])
    ydim.append(couter[1])
plt.bar(xdim, ydim)
#plt.show()



train_data['content1_len'] = train_data['content1'].str.len()
train_data['content2_len'] = train_data['content2'].str.len()
train_data['content3_len'] = train_data['content3'].str.len()
train_data['content4_len'] = train_data['content4'].str.len()
train_data['content5_len'] = train_data['content5'].str.len()

train_data['title1_len'] = train_data['title1'].str.len()
train_data['title2_len'] = train_data['title2'].str.len()
train_data['title3_len'] = train_data['title3'].str.len()
train_data['title4_len'] = train_data['title4'].str.len()
train_data['title5_len'] = train_data['title5'].str.len()


print(train_data['content1_len'].describe())
print(train_data['content2_len'].describe())
print(train_data['content3_len'].describe())
print(train_data['content4_len'].describe())
print(train_data['content5_len'].describe())




print(train_data['title1_len'].describe())
print(train_data['title2_len'].describe())
print(train_data['title3_len'].describe())
print(train_data['title4_len'].describe())
print(train_data['title5_len'].describe())

answer = train_data['answer']
re_ans = r'^\@content([1-5])\@(.*?)\@content([1-5])\@'
import re
regex_ans = re.compile(re_ans)
regex_supp = regex_ans


for d in train_data.iterrows():
    print(d[1]['keyword'])
    
    
    bridging_entity, keyword = d[1]['bridging_entity'], d[1]['keyword']
    content1, content2, content3, content4, content5 = d[1]['content1'], d[1]['content2'], d[1]['content3'], d[1]['content4'], d[1]['content5']
    question, question_id =  d[1]['question'], d[1]['question_id']
    supp_ori = d[1]['supporting_paragraph']
    answer_ori = d[1]['answer']
    title1, title2, title3, title4, title5 = d[1]['title1'], d[1]['title2'],d[1]['title3'],d[1]['title4'],d[1]['title5']

    # answer info extract 
    ans_mat = regex_ans.match(answer_ori)
    assert ans_mat is not None 
    assert  ans_mat.group(1) ==  ans_mat.group(3)
    ans_content_id = ans_mat.group(1)
    answer = ans_mat.group(2)
    print('-' * 10)
    print(ans_content_id)
    print('question is {}\n'.format(question))
    print(answer)

    content = ''
    if ans_content_id == '1': content = content1 
    elif ans_content_id == '2': content = content2 
    elif ans_content_id == '3': content = content3
    elif ans_content_id == '4': content = content4
    elif ans_content_id == '5': content = content5
    else:  raise ValueError
    print(content)

    assert answer in content 
        


    # supporting_paragraph info extract
    supp_mat = regex_supp.match(supp_ori)
    assert supp_mat is not None 
    assert  supp_mat.group(1) ==  supp_mat.group(3)
    supp_content_id = supp_mat.group(1)
    supp = supp_mat.group(2)

    print(supp_content_id)
    print(supp)
    
    #assert answer in supp


    
    



sys.exit(0)


data['content_len'] = pd_data['content'].str.len()


print("\ntitle len desc is:")
print(pd_data['title_len'].describe())
print("\ncontent len desc is:")
print(pd_data['content_len'].describe())


print("content len of 90% is {}".format(pd_data['content_len'].quantile(.9)))
print("title len of 90% is {}".format(pd_data['title_len'].quantile(.9)))


print("content len of 99% is {}".format(pd_data['content_len'].quantile(.99)))
print("title len of 99% is {}".format(pd_data['title_len'].quantile(.99)))




def read_data(data_file, table_file):
    data, tables = [], {}
    with open(data_file) as f:
        for l in f:
            data.append(json.loads(l))
    with open(table_file) as f:
        for l in f:
            l = json.loads(l)
            d = {}
            d['headers'] = l['header']
            d['header2id'] = {j: i for i, j in enumerate(d['headers'])}
            d['content'] = {}
            d['keywords'] = {}
            d['all_values'] = set()
            d['types'] = l['types']
            d['title'] = l['title']
            rows = np.array(l['rows'])
            for i, h in enumerate(d['headers']):
                d['content'][h] = set(rows[:, i])
                if d['types'][i] == 'text':
                    d['keywords'][i] = ''
                    # get_key_words(d['content'][h])
                else:
                    d['keywords'][i] = ''

                d['all_values'].update(d['content'][h])
            # print(d['keywords'])
            d['all_values'] = set([i for i in d['all_values'] if hasattr(i, '__len__')])
            tables[l['id']] = d
    if toy:
        data = data[:toy_data_cnt]
    return data, tables

if mode != 'test':
    train_data, train_tables = read_data(
                            os.path.join(train_data_path, 'train.json'),
                            os.path.join(train_data_path, 'train.tables.json')
                            ) # 41522  5013


valid_data, valid_tables = read_data(
                        os.path.join(valid_data_path, 'val.json'),
                        os.path.join(valid_data_path, 'val.tables.json')
) # 4396 1197
test_data, test_tables = read_data(
                 os.path.join(test_file_path, 'final_test.json'),
                 os.path.join(test_file_path, 'final_test.tables.json')
)
