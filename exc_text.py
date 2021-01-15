
#用来做测试集，other_word里存的是test集有，而训练集没有的单词，清除他们
def read_data_clean(path):
    other_word = open('other_word', 'r', encoding='utf-8').readlines()
    other_word = [i.replace('\n','') for i in other_word]
    with open(path, "rb") as f:
        data = f.read().decode("utf-8")
    train_data = data.split("\n\n")  # 双行切分
    train_data = [token.split("\n") for token in train_data]    #逐个句子切分
    train_data = [[j.split() for j in i] for i in train_data]   #[   [ ['中','B-LOC'],['国','I-LOC'],xxxx ]                ]
    train_data.pop()
    train_data_ = []
    for line in train_data:
        temp = []
        for word,label in line:
            if word not in other_word:
                temp.append([word,label])

        train_data_.append(temp)
    train_x = [[token[0] for token in sentence] for sentence in train_data_] #[  ['中','国','x',xxx],['我','们',xxx]       ]
    train_y = [[token[1] for token in sentence] for sentence in train_data_]

    return train_x,train_y,train_data_

#常规读取
def read_data(path):
    with open(path, "rb") as f:
        data = f.read().decode("utf-8")
    train_data = data.split("\n\n")  # 双行切分
    train_data = [token.split("\n") for token in train_data]    #逐个句子切分
    train_data = [[j.split() for j in i] for i in train_data]   #[   [ ['中','B-LOC'],['国','I-LOC'],xxxx ]                ]
    train_data.pop()    #弹出最后一个回车
    train_x = [[token[0] for token in sentence] for sentence in train_data] #[  ['中','国','x',xxx],['我','们',xxx]       ]
    train_y = [[token[1] for token in sentence] for sentence in train_data]
    return train_x,train_y,train_data

'''
train_x,train_y,train_data =  read_data('data/example.train')
val_x,val_y,val_data = read_data('data/example.dev')
test_x,test_y,test_data = read_data('data/example.test')

all_data_x = train_x+val_x
all_data_x = all_data_x+[['BIN','EOS']]
all_data_y = train_y+val_y+[['O','O']]

x_num,y_num = 1,1
string_id_x={}
string_id_y={}
for index in range(len(all_data_x)):
    for i in range(len(all_data_x[index])):
        char = all_data_x[index][i]
        if char not in string_id_x:
            string_id_x[char] = x_num
            x_num+=1
        label_char = all_data_y[index][i]
        if label_char not in string_id_y:
            string_id_y[label_char] = y_num
            y_num+=1
print(string_id_y)
# print(len(string_id_x))
'''