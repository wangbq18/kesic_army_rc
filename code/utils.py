# 
import pandas as pd 

def write_csv(data_json, target_file):
    """
    input: data_json,
          eg: {'question_id': ['q1','q2'], 'answer': ['中国,美国', '巴基斯坦,印度']}
    """
    header_name = [k for k in data_json]
    df = pd.DataFrame(data_json, columns=header_name)
    df.to_csv(target_file, index=False)



def write_csv_test():
    data_json = {'question_id': ['q1','q2', 'q3'], 'answer': ['中国,美国', '巴基斯坦,印度', 1.23]}
    write_csv(data_json, './test.csv')



def read_csv(csv_file):
    """
    """
    df = pd.read_csv(csv_file)
    answer = df['answer']
    print(answer)
    new = answer.str.split(',')
    print(new)


if __name__ == "__main__":
    # write_csv_test()
    read_csv('./test.csv')

