from Homework2 import data_manager as dm
from sklearn.model_selection import train_test_split
import numpy as np
import math
NUM_CLASS=20# 类的个数

def get_vocab(X_train):
    '''
    从单词数组中，得到相应词汇表
    :param X_train: list, words
    :return: vocab,list, 词汇表 vocabulary
    '''
    vocab=set()
    for doc in X_train:
        vocab=vocab|set(doc)
    return list(vocab)

def words_freq_proc(words_list,frequency):
    '''
    去掉在words_list中出现次数小于frequency的单词，去掉每个文档中的重复单词
    :param words_list: list, words
    :param frequency:
    :return:new_wrods_list，其中没有词频小于frequency的单词，而且每一个文档中的每一个单词只出现一次
    '''
    word_frequency=dict()
    for doc in words_list:
        for word in doc:
            word_frequency[word]=word_frequency.get(word,0)+1
    new_words_list=[]
    for doc in words_list:
        new_list=set()
        for word in doc:
            if(word_frequency[word]>=frequency):
                new_list.add(word)
        new_words_list.append(list(new_list))
    return new_words_list

def naive_bayes(x_test,y_x_prob,y_prob,vocab):
    '''
    伯努利类型朴素贝叶斯分类
    :param x_test: list
    :param y_x_prob: list P(x|y)
    :param y_prob: list P(y)
    :param vocab: list
    :return:
    '''
    res_prob=[0]*NUM_CLASS
    for i in range(NUM_CLASS):
        for j in range(len(vocab)):
            if(x_test.__contains__(vocab[j])):
                res_prob[i]+=math.log(y_x_prob[i][j])
            else:
                res_prob[i]+=math.log(1-y_x_prob[i][j])
        res_prob[i]+=math.log(y_prob[i])
    return np.array(res_prob).argmax()

def calc_y_prob(Y_train):
    '''
    计算P(y)
    :param Y_train:
    :return:
    '''
    y_prob=[1]*NUM_CLASS
    for label in Y_train:
        y_prob[label]+=1
    for i in range(NUM_CLASS):
        y_prob[i]=y_prob[i]/(len(Y_train)+NUM_CLASS)
    return y_prob


def main():
    '''
    计算预测准确率
    :param
    :return:
    '''
    X_train, Y_train, X_test, Y_test = data_load()
    vocab=get_vocab(X_train)
    y_prob=calc_y_prob(Y_train)#计算P(y)
    #y_x_num=[[1]*len(vocab)]*NUM_CLASS#取1的概率   坑爹呀! 二维数组不能这样初始化，如果这样生成二维数组，其每一个元素其实都是指向的同一个位置y_xnum[0]和y_num[1]是指向的同一个位置
    y_x_num=[[1]*len(vocab) for i in range(NUM_CLASS)]#
    y_num=[2]*NUM_CLASS#每个类对应的文档总个数\
    for i in range(len(X_train)):#计算y_x_num,和y_num，用于计算P(xi|y)
        for word in X_train[i]:
            y_x_num[Y_train[i]][vocab.index(word)]+=1# 相应的class里的相应的单词的个数加1
        y_num[Y_train[i]]+=1

    y_x_prob=[]###相应特征取1的概率
    for i in range(NUM_CLASS):#求得概率
        y_x_prob.append(list(np.array(y_x_num[i])/y_num[i]))
    #predict X_test and calculate accuracy of prediction.
    num_right=0
    for i in range(len(X_test)):
        label=naive_bayes(X_test[i],y_x_prob,y_prob,vocab)
        if(label==Y_test[i]):
            num_right+=1
        if((i+1)%10==0):
            print(num_right/(i+1))
    return num_right/len(X_test)


def data_load():
    '''
    读取中间数据:X_train,Y_train,X_test,Y_test
    :return:
    '''
    with open('.\\tmp\\X_train.txt','r') as data:
        tmp=data.read()
        X_train=eval(tmp)
    with open('.\\tmp\\Y_train.txt','r') as data:
        tmp=data.read()
        Y_train=eval(tmp)
    with open('.\\tmp\\X_test.txt','r') as data:
        tmp=data.read()
        X_test=eval(tmp)
    with open('.\\tmp\\Y_test.txt','r') as data:
        tmp=data.read()
        Y_test=eval(tmp)
    return X_train,Y_train,X_test,Y_test

def data_save():
    '''
    保存中间数据:X_train,Y_train,X_test,Y_test
    :return:
    '''
    docs_list, labels_list = dm.doc_load()
    words_list = dm.doc_split(docs_list)
    words_list1 = words_freq_proc(words_list, 15)
    X_train, X_test, Y_train, Y_test = train_test_split(words_list1, labels_list, test_size=0.2, random_state=42)

    with open('.\\tmp\\X_train.txt','w') as data:
        data.write(str(X_train))
    with open('.\\tmp\\Y_train.txt','w') as data:
        data.write(str(Y_train))
    with open('.\\tmp\\X_test.txt','w') as data:
        data.write(str(X_test))
    with open('.\\tmp\\Y_test','w') as data:
        data.write(str(Y_test))

if __name__=="__main__":
    print("begin")
    flag=True
    if(True):
        main()
    else:
        data_save()
        main()


