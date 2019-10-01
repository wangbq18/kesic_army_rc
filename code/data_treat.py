import random 

import re 
import sys 
def mail_test():
    """
    一个解析出邮箱的小测试
    # 括号是用来涵盖输出的值要哪些的
    # @是否转义都可以
    """
    str1 = 'aaf ssa@ss.net asdf  asdb@163.com.cn asdf ss-a@ss.net asdf asdd.cba@163.com afdsaf'
    reg_str1 = r'([\w-]+(\.[\w-]+)*@[\w-]+(\.[\w-]+)+)'
    mod = re.compile(reg_str1)
    items = mod.findall(str1)
    for item in items:
        print(item)

import os 
import sys 
import pandas  as pd 
import jieba 
import editdistance 
import numpy as np 

def get_stop_words():
    file_name = './stopwords.txt'
    stop_word_set = set([])
    with open(file_name, 'r') as f:
        for line in f.readlines():
            stop_word_set.add(line.strip())
    # print(stop_word_set)
    return stop_word_set





stop_words_set = get_stop_words()
# sys.exit(0)


re_ans = r'content([1-5])@(.*?)@content([1-5])+'

regex_ans = re.compile(re_ans)

regex_supp = regex_ans
test_path = '../data/test_data_r0.csv'
train_path = '../data/train_round_0.csv'

ret = regex_supp.findall('@content1@中国@content1@@content2@美国@content2@')
ret = regex_supp.findall('@content1@中国@content1@')
print(ret)

# sys.exit(0)
def text_prepro(text):
    return re.sub('\s', '' ,text)

def most_similar(source, target_list):
    """   
    　这个只能针对于文本进行匹配的
      从词表中找最相近的词（当无法全匹配的时候）
    """
    score_list = [editdistance.eval(source, t) for t in target_list]

    return 
    if len(target_list) == 0:
        return None 
    s_set = set([item for item in source])
    contain_score = []
    un_contain_score = []# target当中相比于source多出来的部分
    for target in target_list:
        t_set = set([t for t in target])
        contain_score.append(len(s_set & t_set))
        # un_contain_score.append(len(t_set.difference(s_set))) #
        un_contain_score.append(0) # 先不扣分了...
    char_match_score = [contain_score[idx]  for idx in range(len(target_list))]

    # 如果最高匹配分数为0,说明一个匹配的都没有，，那么返回None 
    if max(char_match_score) == 0: return None 

    # 下面计算编辑距离分数
    e_d_score = [ len(source) - editdistance.eval(source, t) for t in target_list] 

    final_score = [char_match_score[idx] + e_d_score[idx] for idx in range(len(target_list))]

    return target_list[final_score.index(max(final_score))]

def most_similar_2(w, s):
    """从句子s中找与w最相近的片段，
    借助分词工具和ngram的方式尽量精确地确定边界。
     w : cond value 
     s: question 
     输入和输出的相似度函数不应该相同
     对于输入来说: 进行自动标注的时候，是按照相邻原则来标记的,所以输入采用的相似度方法是n-gram 
     对于输出来说： xxx
    """
    sw = jieba.lcut(s)
    sl = [x for x in list(sw)]
    sl.extend([char for char in s])
    sl.extend([''.join(i) for i in zip(sw, sw[1:])]) # 2-gram 
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:])]) # 3-gram 
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:])]) # 4-grarm 
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:], sw[4:])]) # 5-gram
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:], sw[4:], sw[5:])]) # 6-gram
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:], sw[4:], sw[5:], sw[6:])]) # 7-gram
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:], sw[4:], sw[5:], sw[6:], sw[7:])]) # 8-gram     
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:], sw[4:], sw[5:], sw[6:], sw[7:], sw[8:])]) # 9-gram 
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:], sw[4:], sw[5:], sw[6:], sw[7:], sw[8:], sw[9:])]) # 10-gram 
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:], sw[4:], sw[5:], sw[6:], sw[7:], sw[8:], sw[9:], sw[10:])]) # 11-gram 
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:], sw[4:], sw[5:], sw[6:], sw[7:], sw[8:], sw[9:], sw[10:], sw[11:])]) # 12-gram 
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:], sw[4:], sw[5:], sw[6:], sw[7:], sw[8:], sw[9:], sw[10:], sw[11:], sw[12:])]) # 13-gram 
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:], sw[4:], sw[5:], sw[6:], sw[7:], sw[8:], sw[9:], sw[10:], sw[11:], sw[12:], sw[13:])]) # 14-gram 
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:], sw[4:], sw[5:], sw[6:], sw[7:], sw[8:], sw[9:], sw[10:], sw[11:], sw[12:], sw[13:], sw[14:])]) # 15-gram 
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:], sw[4:], sw[5:], sw[6:], sw[7:], sw[8:], sw[9:], sw[10:], sw[11:], sw[12:], sw[13:], sw[14:], sw[15:])]) # 16-gram 
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:], sw[3:], sw[4:], sw[5:], sw[6:], sw[7:], sw[8:], sw[9:], sw[10:], sw[11:], sw[12:], sw[13:], sw[14:], sw[15:], sw[16:])]) # 12-gram 
    return most_similar(w, sl) 




def get_passages_n_gram(question, input_passage, max_len=180):
    q_len = len(question)
    sentens = input_passage.split('。')
    sentens = [senten for senten in sentens if len(senten.strip()) > 0]
    passages = []
    passages.extend([senten for senten in sentens[:max_len - q_len]])
    # passages.extend(['。'.join(senten) for senten in zip(sentens, sentens[1:])])
    # passages.extend(['。'.join(senten) for senten in zip(sentens, sentens[1:],  sentens[2:])])
    allowed_passage = [passage for passage in passages if len(passage) + q_len <= max_len and  len(passage.strip()) >=1] 
    return allowed_passage




ch_regex = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]')
def get_all_pun_stat():
    '''
    获取所有标点的统计信息
    '''
    train_data = pd.read_csv(train_path)
    pun_d = {}
    for d in  train_data.iterrows():
        # if ids == 100: break 
        # print(d[1]['keyword']) 
        bridging_entity, keyword = d[1]['bridging_entity'], d[1]['keyword']
        content1, content2, content3, content4, content5 = text_prepro(d[1]['content1']), text_prepro(d[1]['content2']), \
                        text_prepro(d[1]['content3']), text_prepro(d[1]['content4']), text_prepro(d[1]['content5'])
        question, question_id =  text_prepro(d[1]['question']), d[1]['question_id']
        supp_ori = text_prepro(d[1]['supporting_paragraph'])
        answer_ori = text_prepro(d[1]['answer'])
        title1, title2, title3, title4, title5 = d[1]['title1'], d[1]['title2'],d[1]['title3'],d[1]['title4'],d[1]['title5']
        content_list = [content1, content2, content3, content4, content5]
        
        for content in content_list:
            paras = content.split('。')
            for para in paras:
                pun_list = ch_regex.findall(para)
                for pun in pun_list:
                    pun_d.setdefault(pun, 0)
                    pun_d[pun] += 1 
                # print(biaodian)
                print(pun_d)

                

    pun_after_sort = sorted(pun_d.items(), key=lambda pun:pun[1], reverse=True)
    print(pun_after_sort)


# get_all_pun_stat()

def get_all_content_contain():
    """
    获取包含xxx报道的数据
    """
    train_data = pd.read_csv(train_path)

    for d in  train_data.iterrows():
        ids += 1 
        # if ids == 100: break 
        # print(d[1]['keyword']) 
        bridging_entity, keyword = d[1]['bridging_entity'], d[1]['keyword']
        content1, content2, content3, content4, content5 = text_prepro(d[1]['content1']), text_prepro(d[1]['content2']), \
                        text_prepro(d[1]['content3']), text_prepro(d[1]['content4']), text_prepro(d[1]['content5'])
        question, question_id =  text_prepro(d[1]['question']), d[1]['question_id']
        supp_ori = text_prepro(d[1]['supporting_paragraph'])
        answer_ori = text_prepro(d[1]['answer'])
        title1, title2, title3, title4, title5 = d[1]['title1'], d[1]['title2'],d[1]['title3'],d[1]['title4'],d[1]['title5']
        content_list = [content1, content2, content3, content4, content5]
        pun_d = {}
        for content in content_list:
            paras = content.split('。')
            for para in paras:
                if '报道' in para:
                    print(para) 

# sys.exit(0)


def get_word_freq():
    """
    获取词频数据,进而对文章进行进一步清洗
    """
    train_data = pd.read_csv(train_path)
    word_dict = {}

    for d in  train_data.iterrows():
        bridging_entity, keyword = d[1]['bridging_entity'], d[1]['keyword']
        content1, content2, content3, content4, content5 = text_prepro(d[1]['content1']), text_prepro(d[1]['content2']), \
                        text_prepro(d[1]['content3']), text_prepro(d[1]['content4']), text_prepro(d[1]['content5'])
        question, question_id =  text_prepro(d[1]['question']), d[1]['question_id']
        supp_ori = text_prepro(d[1]['supporting_paragraph'])
        answer_ori = text_prepro(d[1]['answer'])
        title1, title2, title3, title4, title5 = d[1]['title1'], d[1]['title2'],d[1]['title3'],d[1]['title4'],d[1]['title5']
        content_list = [content1, content2, content3, content4, content5]
        
        for content in content_list:
            paras = content.split('。')
            
            for para in paras:
                words = [w for w in jieba.cut(para)]
                for word in words:
                    word_dict.setdefault(word, 0)
                    word_dict[word] += 1
    print(word_dict)

# get_word_freq()
# sys.exit(0)


# 几种匹配模式
# 据xxx报道
# match1_re = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9“]{0,3}据(.*?)报道$')
# 报道称
match2_re = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9]{0,20}报道[\u4e00-\u9fa5a-zA-Z0-9]{0,4}$')
# 匹配 【】
match3_re = re.compile(r'【(.*?)】')
# （图片来源于网络）
# （图中中国军队士兵正在操作82迫） 含有图的
match4_re = re.compile(r'\(记者(.*?)\)')
para_regex = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9，。！？,.!?、/\-；（）%()～~]')
def text_pre_treat(text):
    """
    文本预处理
    """
    para = text 
    para = para.replace('[', '【')
    para = para.replace(']', '】')
    para = para.replace('（', '(')
    para = para.replace('）', ')')# 中文括号统一转英文括号
    para = para.replace('”', '') # 中文双引号
    para = para.replace('“', '')
    para = para.replace('(图片来源于网络)', '')
    para = para.replace('(上图)', '')

    para = para.replace('２', '2')
    para = para.replace('５', '5')
    para = para.replace('％', '%')
    para = para.replace('1', '1')
    para = para.replace('６', '6')
    para = para.replace('Слава', 'cnbba')

    

    # para = para.replace('《', '')
    # para = para.replace('》', '')
    #新增 对para做预处理
    para_first = re.split('，|,|；|;', para)[0]
    if match2_re.findall(para_first):
        # para = para.replace(para_first, '')
        regex = '{}[，|,|；|;]'.format(para_first)
        try:
            para = re.sub(regex, '', para)
        except:
            pass 
            # print('excep ----')
            # print(para)
            # raise 
    # print('-' * 10) 
    # print(para)
    # print(match3_re.findall(para))
    if match3_re.findall(para):
        para_back = para
        for item in match3_re.findall(para_back):
            # print(item)
            regex = '【{}】'.format(item)
            # print('regex {}'.format(regex))
            para = para.replace(regex, '')
    if match4_re.findall(para):
        for m in match4_re.findall(para):
            # print(m)
            para = para.replace('(记者{})'.format(m), '')
        
    para = ''.join(para_regex.findall(para))  
    return para 
    



def get_latent_para(question, content_list, max_len=510, mat_supps=None):
    """
    从content_list中获取潜在的可能包含答案的段落, 
      要保证content顺序性服从
    ret: 
       用于进行训练的文本数据, 默认截断长度为510 
    
    """

    all_paras_sim_stat = {}
    question = text_pre_treat(question)
    question_set = set([word for word in jieba.cut(question)])
    question_set -= stop_words_set
    print('-' * 10)


    for c_idx, content in enumerate(content_list):
        paras = content.split('。') # 有没有必要再通过感叹号来进行段落划分
        # 标点符号没有参与的意义
        for pid, para in enumerate(paras):
            # print(para)
            para = text_pre_treat(para)
        
            # 新增停止
            para_set = set([word for word in jieba.cut(para)]) - stop_words_set
            # all_paras_sim_stat[para] = '{}:{}'.format(idx, len(question_set & para_set)) 
            all_paras_sim_stat[para] = [len(question_set & para_set), pid, c_idx]
    # all_paras_sim_stat = sorted(all_paras_sim_stat.items(), key=lambda item: item[1])
    
    para_after_sort = sorted(all_paras_sim_stat.items(), key=lambda all_paras_sim_stat:all_paras_sim_stat[1][0], reverse=True)
    import_para = '。'.join([para[0] for para in para_after_sort])[:max_len - len(question)]

    # 看看和重点语句的交集: 
    # import_set_and = set(import_para.split('。')) & mat_supps
    # print('and and is {}'.format(len(import_set_and)))
    # print(import_para)
    return import_para, para_after_sort # 如果不需要反转

    # 是否有必要反转? 
    reserverd_para = para_after_sort[:len(import_para.split('。'))]
    reserverd_para_sort =  sorted(reserverd_para, key=lambda item:str(item[1][2]) + ':' + str(item[1][1]),  
    reverse=False)

    import_para = '。'.join([para[0] for para in reserverd_para_sort])[:max_len - len(question)]
    return import_para, question
    # print(import_para)
    # final_list  = [] 
    # for content in content_list:
    #     paras = content.split('。') # 有没有必要再通过感叹号来进行段落划分
    #     # 标点符号没有参与的意义
    #     for para in paras:
    #         para = text_pre_treat(para)
    #         for import_para in import_para.split('。'):
    #             if para == import_para:
    #                 final_list.append(para)
    #             # elif import_para in para:
    #             #     final_list.append(para)
    # print('。'.join(final_list))
    # return '。'.join(final_list)
    # print('-' * 10)
    # print(question)
    # print(para_after_sort) 
    # para_after_sort = set([para[0] for para in para_after_sort])

    ## 还原最初的顺序
    # print(import_para)
    # print(question)
    # print(len(import_para) + len(question))
    return import_para, para_after_sort

def check_train_support_and_brid():
    '''现在需要的就是 passage , question, answer 单独搞出来,
       另外一种方式, 长度控制在256,其中开头由支撑段落预处理后填充,然后后面用通过算法搞出来的   相似句子进行填充

       要有正例, 有反例, 训练的时候,如何制造反例??不含有支撑段落的，但是却很相似的
    '''
    train_data = pd.read_csv(train_path)
    data_json = []
    idx = 0
    passages_train = {}

    content_supp_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for d in  train_data.iterrows():
        bridging_entity, keyword = d[1]['bridging_entity'], d[1]['keyword']
        content1, content2, content3, content4, content5 = text_prepro(d[1]['content1']), text_prepro(d[1]['content2']), \
                        text_prepro(d[1]['content3']), text_prepro(d[1]['content4']), text_prepro(d[1]['content5'])
        question, question_id =   text_pre_treat(d[1]['question']), d[1]['question_id']
        supp_ori = text_prepro(d[1]['supporting_paragraph'])
        answer_ori = text_prepro(d[1]['answer'])
        title1, title2, title3, title4, title5 = d[1]['title1'], d[1]['title2'],d[1]['title3'],d[1]['title4'],d[1]['title5']

        # answer info extract 
        ans_mat = regex_ans.findall(answer_ori)
        ans_mat = [ text_pre_treat(ans[1]) for ans in ans_mat]
        # print('*' * 10)
        # print(question)
        # print(ans_mat)
    
        assert len(ans_mat) != 0

        content_list = [content1, content2, content3, content4, content5]
        # import_para = get_latent_para_ver2(question, content_list) # 获取所有材料中的重点语句
        # 答案和对应的支撑段落匹配起来,重点要看看是否可以找到蛛丝马迹,即问题在支撑段落中是否出现
        # 看看supporting graph中有多少包含我们所要的答案,首先解析出来
        content_supp_cnt[supp_ori.count('@content')] += 1 
        mat_supps = regex_supp.findall(supp_ori)
        mat_supps = [text_pre_treat(mat_supp[1]) for mat_supp in mat_supps]
        print(mat_supps)


# check_train_support_and_brid()
# sys.exit(0)

def get_latent_para_ver2(question, content_list):
    """
    从content_list中获取潜在的可能包含答案的段落, 添加一个递归把候选项目放到问题集合中的方式
    ret: 
       用于进行训练的文本数据, 默认截断长度为1024
    """
    import_para, para_after_sort = get_latent_para(question, content_list)
    import_para, para_after_sort = get_latent_para(question + para_after_sort[0][0] + para_after_sort[1][0], content_list)

    return import_para 




stat_info = [0, 0]
ch_good_regex = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9\-“）]')


def mark_answer_in_para(answer_list, para):
    """
    在段落中标记出答案的位置
    input: 
      answer: type: List 
      para: 重点段落
      
    ret: start_pos: the start pos of answer 
         end_pos: the end pos of answer 
    """
    # print(answer_list)
    para = ''.join(ch_good_regex.findall(para))
    for answer in answer_list: 
        ans = ''.join(ch_good_regex.findall(answer[1]))
        if ans in para:
            stat_info[1] += 1 
        else: 
            print('-' * 10)
            print(para)
            print(ans)
            stat_info[0] += 1 
            print(stat_info)
    
        # 可以看看,有多少answer,是在我们的question中的. 


def remove_unwant(answer):
    ans_treat = ''
    start, end = 0, len(answer) 
    for word in answer: 
        if not ch_good_regex.findall(word): start += 1 
        else: break 
    for word in answer[::-1]:
        if not ch_good_regex.findall(word): end -= 1 
        else: break 
    return answer[start:end]


def mark_answer_in_para_new(answer_list, mat_supps, para):
    """
    在段落中标记出答案的位置
    input: 
      answer_list: type: List 
      mat_supps: 关联的重点段落
      para: 重点段落
      
    ret: start_pos: the start pos of answer 
         end_pos: the end pos of answer 

         [{'answer': xx, 'start_pos':, 'end_pos': xx}]
    """
    # print(answer_list)
    # para = ''.join(ch_good_regex.findall(para))
    ret_list = []
    for answer in answer_list: 
        if len(answer.strip()) == 0: 
            print(answer_list)
            raise 
        # ans = ''.join(ch_good_regex.findall(answer[1]))
        # ans = remove_unwant(answer[1])
        answer = answer.replace('。', '，')# 改下标点
        if len(answer.strip()) != 0 and  answer in para: # answer可能出现中间包含句号的情况, answer作为一个整体 
            # para.index from which idx ??? 
            supp_para_start = None 
            for mat_supp in mat_supps:
                # mat_supp可能还可以拆分
                # for mat_item in mat_supp.split('。'):
                mat_supp = remove_unwant(mat_supp)
                if len(mat_supp.strip()) == 0: continue
                answer = remove_unwant(answer)
                if answer in mat_supp and mat_supp in para:
                    supp_para_start = para.index(mat_supp)
                    break 
        


            start_idx = para.index(answer, supp_para_start) # 
            end_idx = start_idx + len(answer)
            
            ret_item = {'answer': answer, 'start_pos': start_idx, 'end_pos': end_idx - 1}
            ret_list.append(ret_item)
    return ret_list 
        #     stat_info[1] += 1 
        # else: 
        #     print('-' * 10)
        #     print(para)
        #     print(ans)
        #     stat_info[0] += 1 
        #     print(stat_info)



def get_test_dataset(test_path):
    test_data_json = [] 
    test_data = pd.read_csv(test_path)
    for d in test_data.iterrows():
        content1, content2, content3, content4, content5 = text_prepro(d[1]['content1']), text_prepro(d[1]['content2']), \
                        text_prepro(d[1]['content3']), text_prepro(d[1]['content4']), text_prepro(d[1]['content5'])
        question, question_id =  text_pre_treat(d[1]['question']), d[1]['question_id']
        title1, title2, title3, title4, title5 = d[1]['title1'], d[1]['title2'],d[1]['title3'],d[1]['title4'],d[1]['title5']
        # test_data_json.append({"question_id": question_id, "question": question, })
        content_list = [content1, content2, content3, content4, content5]
        import_para, _ = get_latent_para(question, content_list, max_len=2000, mat_supps=None) # 从这3000个当中找出值来补充长度
        import_para_list = import_para.split('。')

        # 前几个句子是最重要的了
        for i in range(30):
            import_para_list_tmp = import_para_list
            random.shuffle(import_para_list_tmp)
        
            data_item = {'question': question, 'para': '。'.join(import_para_list_tmp)[:300 - len(question_id)]}

            print(data_item)
            test_data_json.append(data_item)
    np.save('kesic_test_new_method_300.npy', test_data_json)


# get_test_dataset(test_path)

# sys.

def get_test_dataset_new(test_path, max_len=508, pool_str_len=3500, per_sample=20):
    """
    max_len: 问题+段落的最大长度
    pool_str_len:候选字符串长度
    per_sample :

    """
    test_data_json = [] 
    test_data = pd.read_csv(test_path)
    for d in test_data.iterrows():
        content1, content2, content3, content4, content5 = text_prepro(d[1]['content1']), text_prepro(d[1]['content2']), \
                        text_prepro(d[1]['content3']), text_prepro(d[1]['content4']), text_prepro(d[1]['content5'])
        question, question_id =  text_pre_treat(d[1]['question']), d[1]['question_id']
        title1, title2, title3, title4, title5 = d[1]['title1'], d[1]['title2'],d[1]['title3'],d[1]['title4'],d[1]['title5']
        # test_data_json.append({"question_id": question_id, "question": question, })
        content_list = [content1, content2, content3, content4, content5]
        import_para, _ = get_latent_para(question, content_list, max_len=pool_str_len, mat_supps=None) # 从这3000个当中找出值来补充长度
        import_para_list = [ para for para in  import_para.split('。') if len(para.strip()) != 0]

        # 第一个句子万万是要留着的
        first_import_para = import_para_list[0]

        other_import_para = import_para_list[1:]

        # 前几个句子是最重要的了
        for i in range(per_sample):
            test_input_item = [first_import_para]
            random.shuffle(other_import_para)  
            test_input_item.extend(other_import_para)

            # 再按照句号切分一次,然后再shuffle,为了避免第一个重要句子总是出现在句子首
            # print(test_input_item)
            # print('-'  *10)
            para_input_contain_first = '。'.join(test_input_item)[:max_len - len(question)] # 最重要的句子在句首 
            para_input_list = [ para for para in  para_input_contain_first.split('。') if len(para.strip()) != 0]
            random.shuffle(para_input_list)
            find_test_para = '。'.join(para_input_list)
            
            # print(len(question) , len(find_test_para))
            find_test_para  +=  '。' * (max_len - len(question) - len(find_test_para)) 
            # print(find_test_para)
            # print(len(question) + len(find_test_para))
            assert len(question) + len(find_test_para) == max_len 
            # print(find_test_para)
            data_item = {'question': question, 'para': find_test_para, 'question_id': question_id}
            test_data_json.append(data_item)
    np.save('kesic_test_new_method_samp{}_of_{}_len{}.npy'.format(per_sample, pool_str_len, max_len), test_data_json)

# print('start create')

get_test_dataset_new(test_path, max_len=380, pool_str_len=2250, per_sample=12)

sys.exit(0)


def check_train_data():
    data_path = './new_train_method_300.npy'
    datas = np.load(data_path, allow_pickle=True).tolist()
    no_answer_cnt = 1 
    has_answer_cnt = 1 
    q_cnt_info = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for d in datas:
        print(d)
        # print(d['q_cnt'])
        if d['answer_mark']:
            has_answer_cnt += 1 
        else:
            no_answer_cnt += 1 
        q_cnt_info[d['q_cnt']] += 1 
        if d['q_cnt'] == 2 and d['question'].count('？') == 1:
            # print(d['question'])
            pass 

        # print(q_cnt_info)
        # print(has_answer_cnt, no_answer_cnt) # 73483 74530


# check_train_data()
# sys.exit(0)


import random 
def get_train_dataset_new(max_len=380):
    """
    1:1 混合正例与反例    要知道，句子当中可能是没有答案的
    混合的时候，直接使用支撑文案，加一些相似属性，作为我们的训练语语料，长度可以用250-300???  如何再进一步预处理下, 可以设置三次滑动窗口，这样可以滑动到700左右个字符串
    当然如何滑动是个问题？？？毕竟前面的几个句子数据量是最大的. 可以将中间的100多替换掉咋的,多尝试,最大搜索长度为1000 

    支撑段落必须要散落一点
    """
    train_data = pd.read_csv(train_path)
    data_json = []
    idx = 0
    passages_train = {}

    content_supp_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    common_cnt = [0] * 80
    ids = 0 
    import_in_cnt, import_in_cnt_new = 0, 0 
    supp_in_content_cnt = 0 
    un_cnt = 0 
    for d in  train_data.iterrows():
        ids += 1 
        # if ids == 100: break 
        # print(d[1]['keyword'])
        
        bridging_entity, keyword = d[1]['bridging_entity'], d[1]['keyword']
        content1, content2, content3, content4, content5 = text_prepro(d[1]['content1']), text_prepro(d[1]['content2']), \
                        text_prepro(d[1]['content3']), text_prepro(d[1]['content4']), text_prepro(d[1]['content5'])
        question, question_id =   text_pre_treat(d[1]['question']), d[1]['question_id']
        supp_ori = text_prepro(d[1]['supporting_paragraph'])
        answer_ori = text_prepro(d[1]['answer'])
        title1, title2, title3, title4, title5 = d[1]['title1'], d[1]['title2'],d[1]['title3'],d[1]['title4'],d[1]['title5']
        content_list = [content1, content2, content3, content4, content5]

        #  看看支撑段落是否可以在训练样本中找到
        '''
        mat_supps = regex_supp.findall(supp_ori)
        for mat_supp in mat_supps:
            assert mat_supp[0] == mat_supp[2]
            cid, supp_text_treat, supp_text = mat_supp[0], text_pre_treat(mat_supp[1]), mat_supp[1]
            if  remove_unwant(supp_text_treat) not in text_pre_treat(content_list[int(cid) - 1]): #remove unwanted很重要
                print('-' * 10)
                print(supp_text_treat)
                print(supp_text)
                print(cid)
                print(text_pre_treat(content_list[int(cid) - 1]))
                un_cnt +=1 
            # print(un_cnt)
        # mat_supps = [text_pre_treat(mat_supp[1]) for mat_supp in mat_supps]
        continue 
        '''
        # answer info extract 
        ans_mat_ori = regex_ans.findall(answer_ori)
        # print(ans_mat)
        ans_mat = [ remove_unwant(text_pre_treat(ans[1])) for ans in ans_mat_ori]
        # 获取ans_mat对应的supp_mat,按照idx对齐
        for ans in ans_mat:
            # print(ans_mat)
            if  len(ans.strip()) == 0: 
                print(ans_mat)
                print(answer_ori)
                raise 

        assert len(ans_mat) != 0
        content_list = [content1, content2, content3, content4, content5]     
        # 答案和对应的支撑段落匹配起来,重点要看看是否可以找到蛛丝马迹,即问题在支撑段落中是否出现
        # 看看supporting graph中有多少包含我们所要的答案,首先解析出来
        content_supp_cnt[supp_ori.count('@content')] += 1 
        mat_supps_ori = regex_supp.findall(supp_ori)
        mat_supps = [remove_unwant(text_pre_treat(mat_supp[1])) for mat_supp in mat_supps_ori]
   
        import_para, _ = get_latent_para(question, content_list, max_len=3000, mat_supps = mat_supps) # 从这3000个当中找出值来补充长度
        import_para_ori = import_para
        for mat_supp in mat_supps:
            for item in mat_supp.split('。'): # 防止一个支撑段落中含有多个句子
                import_para = import_para.replace(item, '')

        # 制作2个反例
        q_id = question_id
        for i in range(2):
            question_id = '{}_neg_{}'.format(q_id, i)
            import_para_list =  [ para for para in  import_para.split('。') if len(remove_unwant(para.strip())) != 0]
            random.shuffle(import_para_list)
            import_para_input = '。'.join([para for para in  import_para_list if len(remove_unwant(para)) > 0])[:max_len - len(question)]
            # print(mat_supps)
            q_cnt = 1 if len(ans_mat) > 2 else len(ans_mat)
            print(len(import_para_input) + len(question))
            if  len(import_para_input) + len(question) < 380:
                import_para_input = import_para_input +  '。' * (380 - len(import_para_input) - len(question))
            assert len(import_para_input) + len(question) <= 380 and  len(import_para_input) + len(question) >= 378
            assert len(import_para_input) + len(question) == 380
            # print('neg')
            # print(import_para_input)
            passages_train[question_id] = {'question': question, 'para': import_para_input, 'answer_mark': None, 'ans_cnt': 0, 'q_cnt': q_cnt}
            # print(passages_train[question_id])
        for i in range(2): # 二个正例样本
            question_id = '{}_pos_{}'.format(q_id, i)
            # print(question_id)
            import_para_list = [ para for para in  import_para.split('。') if len(para.strip()) != 0]
            random.shuffle(import_para_list)
            supp_para_len, supp_para_concat = 0, ''
            
            for mat_supp in mat_supps:
                mat_supp = mat_supp.replace('。', '，') #句号改成逗号，让多个句子构成的段落成为整体
                supp_para_len += len(mat_supp)
                supp_para_concat += mat_supp + '。'
            left_part_max_len = 0  if max_len - supp_para_len - len(question)  < 0 else max_len - supp_para_len - len(question) - 1 

            import_para_con = '。'.join([para for para in  import_para_list if len(remove_unwant(para))> 0])[:left_part_max_len] + '。' + supp_para_concat[:max_len - len(question)]
            import_para_con_list = [para_con for para_con in import_para_con.split('。') if len(para_con.strip()) !=0]
            random.shuffle(import_para_con_list)

            import_para_input = remove_unwant('。'.join(import_para_con_list))[:max_len - len(question)]
            # print('pos')
            # print(supp_para_concat)
            # print(import_para_input)
            ans_ret_item = mark_answer_in_para_new(ans_mat, mat_supps, import_para_input)  # ans也有可能是个数组
            if len(ans_mat) != len(ans_ret_item):
                continue 
                # ans_ret_item = mark_answer_in_para_new(ans_mat, mat_supps, import_para_input)
                # print('-' * 10)
                # print(ans_mat)
                # print(ans_ret_item)
                # print(mat_supps)
                # print(import_para_input)
                # raise 
            # print(mat_supps)
            q_cnt = 1 if len(ans_mat) > 2 else len(ans_mat)
            print(len(import_para_input) + len(question))
            print(import_para_input)
            if  len(import_para_input) + len(question) < 380:
                import_para_input = import_para_input +  '。' * (380 - len(import_para_input) - len(question))
            assert len(import_para_input) + len(question) <= 380 and  len(import_para_input) + len(question) >= 378 # 10 <= 380     11> 375
            assert len(import_para_input) + len(question) == 380
            passages_train[question_id] = {'question': question, 'para': import_para_input, 'answer_mark': ans_ret_item, 'ans_cnt': len(ans_ret_item), 'q_cnt': q_cnt}
            # print(passages_train[question_id])
        # print(len(passages_train))

        # 查看这些标记数据 是否包含在里面呢

        # import_para_set = set(import_para.split('。'))
        mode = 'check_supps_in_content_un'
    
        if mode == 'check_supps_in_content': # 查看支撑段落是否在内容集合里面
            mat_supps_single = []
            for mat_supp in mat_supps:
                mat_supps_single.extend(mat_supp.split('。'))
            mat_supps_set = set(mat_supps_single)   
            if '' in mat_supps_set:
                mat_supps_set.remove('')
            for m in mat_supps_set:
                if len(m.strip()) == 0 :raise 
            all_para_set = set([])        
            for c_idx, content in enumerate(content_list):
                paras = content.split('。') # 有没有必要再通过感叹号来进行段落划分
                for pid, para in enumerate(paras):
                    # print(para)
                    para = text_pre_treat(para)
                    all_para_set.add(para)
            if len(all_para_set & mat_supps_set) == len(mat_supps_set):
                supp_in_content_cnt += 1  
            else:
                print('*' * 10)
                print(mat_supps_set)
                # print(all_para_set & mat_supps_set)
                print(all_para_set)

            print('In cnt is {}'.format(supp_in_content_cnt))  # 19021
            # 虽然这个19021距离所有样本差距很多，但是注意一点，这个支撑段落可能只是一个句子中的一部分，所以会出现这么多匹配不上的
        elif mode == 'check_ans_in_supp': # 查看答案是否在支撑段落中可以找到
            pass 
        # print(len(passages_train))
    passages_train_list = [passages_train[k] for k in passages_train.keys()]
    np.save('./new_train_method_380.npy', passages_train_list)


get_train_dataset_new()


sys.exit(0)
def get_train_dataset(treat_mode = 'bert', max_len=384):
    ## 现在需要的就是 passage , question, answer 单独搞出来
    train_data = pd.read_csv(train_path)
    data_json = []
    idx = 0
    passages_train = {}

    content_supp_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    common_cnt = [0] * 80
    ids = 0 
    for d in  train_data.iterrows():
        ids += 1 
        # if ids == 100: break 
        # print(d[1]['keyword'])
        
        bridging_entity, keyword = d[1]['bridging_entity'], d[1]['keyword']
        content1, content2, content3, content4, content5 = text_prepro(d[1]['content1']), text_prepro(d[1]['content2']), \
                        text_prepro(d[1]['content3']), text_prepro(d[1]['content4']), text_prepro(d[1]['content5'])
        question, question_id =   text_pre_treat(d[1]['question']), d[1]['question_id']
        supp_ori = text_prepro(d[1]['supporting_paragraph'])
        answer_ori = text_prepro(d[1]['answer'])
        title1, title2, title3, title4, title5 = d[1]['title1'], d[1]['title2'],d[1]['title3'],d[1]['title4'],d[1]['title5']

        # answer info extract 
        ans_mat = regex_ans.findall(answer_ori)
        print(ans_mat)
        for ans in ans_mat:
            if  len(ans[1].strip()) == 0: 
                print(ans_mat)
                print(answer_ori)
                raise 
        ans_mat = [ text_pre_treat(ans[1]) for ans in ans_mat]
        
        for ans in ans_mat:
            print(ans_mat)
            if  len(ans.strip()) == 0: 
                print(ans_mat)
                print(answer_ori)
                raise 

        assert len(ans_mat) != 0

        content_list = [content1, content2, content3, content4, content5]
        # import_para = get_latent_para_ver2(question, content_list) # 获取所有材料中的重点语句


        '''
        ans_content_id = int(ans_mat.group(1))
        answer = text_prepro(ans_mat.group(2))

        punctuation = ['。', '？', '，', '.', '、', '”', '.', '“', "：", ',', ':', '！']
        answer = answer.lstrip(''.join(punctuation))
        answer = answer.rstrip(''.join(punctuation))
        '''
        
        # 答案和对应的支撑段落匹配起来,重点要看看是否可以找到蛛丝马迹,即问题在支撑段落中是否出现
        # print('-' * 10)        
        # print(question)
        # print(ans_mat)
        # 看看supporting graph中有多少包含我们所要的答案,首先解析出来
        content_supp_cnt[supp_ori.count('@content')] += 1 
        mat_supps = regex_supp.findall(supp_ori)
        mat_supps = [text_pre_treat(mat_supp[1]) for mat_supp in mat_supps]
        # continue 
        # 看看 支撑段落 和 问题之间 交集
        '''
        question_set = set([word for word in jieba.cut(question)])
        question_set -= stop_words_set
        for mat_supp in mat_supps:
            # print(mat_supp[1])
            word_sets = set([word for word in jieba.cut(mat_supp[1])])
            para_set = word_sets - stop_words_set
            if len(question_set & para_set) == 0:
                print('-' * 10)
                print(question)
                print(mat_supp[1])
                print(mat_supps)
            # print(len(question_set & para_set))
            common_cnt[len(question_set & para_set)] += 1 
        '''

        '''
        for content in content_list:
            paragraphs = content.split('。')
            for para in paragraphs:
                word_sets = set([word for word in jieba.cut(para)])
                print(word_sets - stop_words_set)
        '''
        if treat_mode == 'bert':
            max_len = 506
            import_para, answer = get_latent_para(question, content_list, max_len=max_len, mat_supps = mat_supps) # from 320 to 506 

            ret_item = mark_answer_in_para_new(ans_mat, import_para)  # ans也有可能是个数组
            # print(question)
            # print(ret_item)
            if ret_item:
                passages_train[question_id] = {'question': question, 'para': import_para, 'answer_mark': ret_item}
                # print(ret_item)
                #print(passages_train_bert[question_id] )
            else: 
                print('-' * 10)
                print(question_id)
                print(question)
                print(ans_mat)
                # print(import_para)
                new_import_para = '。'.join(mat_supps)  + import_para
                new_import_para = new_import_para[:max_len - len(question)]

                # print(new_import_para)
                ret_item = mark_answer_in_para_new(ans_mat, new_import_para)
                if ret_item:
                    passages_train[question_id] = {'question': question, 'para': new_import_para, 'answer_mark': ret_item}
                    print(passages_train[question_id])
                    assert len(new_import_para) + len(question) == max_len
            print(len(passages_train))
            '''

            for idx, content in enumerate(content_list): 
                clean_passages = get_passages_n_gram(question, content_list[idx], max_len=1024) 
            for clean_passage in clean_passages:
                train_answer = answer if idx == ans_content_id - 1 and answer in clean_passage else '' 

                passages_train_bert.append({"answer": train_answer, 'passage': clean_passage, 'question': question, 'id': question_id})
            '''

        elif treat_mode == 'dgcnn_old':
            # 按照dgcnn格式处理数据
            passages_train  = [] 
            for idx, content in enumerate(content_list): 
                clean_passages = get_passages_n_gram(question, content_list[idx]) 
                # judege supporting_paragraph in the clean passages 
                print(clean_passages)
                print('\n' * 3)
                continue 
                # 
                for clean_passage in clean_passages:
                    train_answer = answer if idx == ans_content_id - 1 and answer in clean_passage else '' 
                    passages_train.append({"answer": train_answer, 'passage': clean_passage})
            if len(passages_train) == 0: 
                print(content_list)
                print('-----')
                sys.exit(0)
            data_json.append({'question': question, 'id': question_id,  'passages': passages_train})

        elif treat_mode == 'dgcnn_new':
            passages_train  = [] 
            clean_passage = get_latent_para(question, content_list, max_len=320)[0]
            if answer in clean_passage:
                train_answer = answer 
            else: 
                continue 
                passages_train.append({"answer": train_answer, 'passage': clean_passage})
            if len(passages_train) == 0: 
                print(content_list)
                print('-----')
                sys.exit(0)
            data_json.append({'question': question, 'id': question_id,  'passages': passages_train})


    # find_set = set([])
    # all_set = set([])
    # for passage in passages_train:
    #     all_set.add(passage['id'])
    #     if passage['answer'] != '':
    #         find_set.add(passage['id'])
    # print(len(all_set - find_set))

    if  treat_mode == 'bert':
        import numpy as np 
        # print(passages_train)
        
        np.save('kesic_508.npy', passages_train)
    elif treat_mode == 'dgcnn':
        pass 
    sys.exit(0)


    find_set = set([])
    all_set = set([])
    for passage in passages_train_bert:
        all_set.add(passage['id'])
        if passage['answer'] != '':
            find_set.add(passage['id'])
    print(len(all_set - find_set))
    sys.exit(0)



    has_answer_pass = [passage for passage in passages_train_bert if passage['answer'] != '']
    print(len(passages_train_bert))
    print(len(has_answer_pass))
    # print(passages_train_bert[:100])
    import numpy as np 
    # np.save('kesic_bert.npy', passages_train_bert)

    sys.exit(0)


    np.save('kesic.npy', data_json)  


     
        
if __name__ == "__main__":
    
    get_train_dataset() 
  



        

