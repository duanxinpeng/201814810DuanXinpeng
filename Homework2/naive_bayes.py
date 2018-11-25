from Homework2 import data_manager as dm
from sklearn.model_selection import train_test_split
import numpy as np
import math
NUM_CLASS=20# 类的个数

def get_vocab(X_train):
    '''
    从单词数组中，得到相应词汇表,并去除训练数据集中的重复单词
    :param X_train: list, words
    :return: vocab,list, 词汇表 vocabulary
    :return: new_list list  去掉重复单词的X_train
    '''
    vocab=set()
    new_list=[]
    for doc in X_train:
        vocab=vocab|set(doc)
        new_list.append(list(set(doc)))
    return list(vocab),new_list

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

def calc_x_y_prob(X_train,Y_train,vocab):
    '''
    计算P(x|y)
    :param X_train: list
    :param Y_train: list
    :param vocab: list
    :return: list
    '''
    x_y_num=[[1]*len(vocab) for i in range(NUM_CLASS)]#某个类中，出现某个单词的文档个数，初始化为1，是拉普拉斯平滑需要
    y_num=[2]*NUM_CLASS#每个类中的文档总个数\，初始化为2，是拉普拉斯平滑需要
    for i in range(len(X_train)):#计算y_x_num,和y_num，用于计算P(xi|y)
        for word in X_train[i]:
            x_y_num[Y_train[i]][vocab.index(word)]+=1# 相应的class里的相应的单词的个数加1
        y_num[Y_train[i]]+=1

    x_y_prob=[]###某个单词存在的概率； 不存在的概率与存在概率互斥，所以直接减一即可
    for i in range(NUM_CLASS):#求得概率
        x_y_prob.append(list(np.array(x_y_num[i])/y_num[i]))
    return x_y_prob

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



def main():
    '''
    计算预测准确率
    :param
    :return:
    '''
    X_train, Y_train, X_test, Y_test = data_load()
    vocab,X_train=get_vocab(X_train)#
    y_prob=calc_y_prob(Y_train)#计算P(y)
    #y_x_num=[[1]*len(vocab)]*NUM_CLASS#取1的概率   坑爹呀! 二维数组不能这样初始化，如果这样生成二维数组，其每一个元素其实都是指向的同一个位置y_xnum[0]和y_num[1]是指向的同一个位置
    x_y_prob=calc_x_y_prob(X_train,Y_train,vocab)
    #predict X_test and calculate accuracy of prediction.
    num_right=0
    for i in range(len(X_test)):
        label=naive_bayes(X_test[i],x_y_prob,y_prob,vocab)
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
    with open('.\\tmp\\Y_test','r') as data:
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
    #words_list1 = words_freq_proc(words_list, 15)
    X_train, X_test, Y_train, Y_test = train_test_split(words_list, labels_list, test_size=0.2, random_state=42)

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
    flag=True#中间数据存在与否
    if(True):
        main()
    else:
        data_save()
        main()


