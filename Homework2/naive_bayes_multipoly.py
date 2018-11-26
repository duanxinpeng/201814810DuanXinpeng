from Homework2 import data_manager as dm
from sklearn.model_selection import train_test_split
import numpy as np
import math
NUM_CLASS=20# 类的个数

def calc_x_y_prob(x_train,y_train):
    '''
    计算P(x|y)
    :param x_train: list
    :param y_train: list
    :return: res：dict list 20个类的单词的概率
    :return: num_all: list 20个类中单词总数，用于拉普拉斯平滑
    '''
    res=[{} for i in range(NUM_CLASS)]
    num_all=[0]*NUM_CLASS
    num_docs=len(x_train)
    for i in range(num_docs):
        for word in x_train[i]:
            res[y_train[i]][word]=(res[y_train[i]].get(word,0)+1)
            num_all[y_train[i]]+=1
    for i in range(NUM_CLASS):
        for key in res[i]:
            res[i][key]=(res[i][key]+1)/(num_all[i]+len(list(res[i].keys())))#加上了拉普拉斯平滑
    return res,num_all

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

def naive_bayes(x_test,x_y_prob,y_prob,num_all):
    '''
    朴素贝叶斯分类，多项式模型
    :param x_test: list
    :param x_y_prob: dict list P(x|y)
    :param y_prob: list P(y)
    :param num_all list 平滑参数,用于拉普拉斯平滑
    :return: predict label
    '''
    res_prob=[0]*NUM_CLASS
    for i in range(NUM_CLASS):
        for word in x_test:
            tmp_prob=x_y_prob[i].get(word,0)
            if(tmp_prob!=0):
                res_prob[i]+=math.log(tmp_prob)
            else:
                res_prob[i]+=math.log(1/(num_all[i]+len(list(x_y_prob[i].keys()))))#加拉普拉斯平滑和不加拉普拉斯平滑的准确率差距之比是0.88比0.005？
        res_prob[i]+=math.log(y_prob[i])
    return np.array(res_prob).argmax()


def main():
    '''
    计算预测准确率
    :param
    :return:
    '''
    X_train, Y_train, X_test, Y_test = data_load()
    #vocab=get_vocab(X_train)
    y_prob=calc_y_prob(Y_train)#计算P(y)
    #y_x_num=[[1]*len(vocab)]*NUM_CLASS#取1的概率   坑爹呀! 二维数组不能这样初始化，如果这样生成二维数组，其每一个元素其实都是指向的同一个位置y_xnum[0]和y_num[1]是指向的同一个位置
    #y_x_num,y_num=y_word_freq(X_train,Y_train)#每个类中所有文档中的所有单词的词频和每个类中所有文档中的所有单词的总和
    x_y_prob,num_all=calc_x_y_prob(X_train,Y_train)
    num_right=0
    x_label=[]
    y_label=[]
    pred_label=[]
    for i in range(len(X_test)):
        label=naive_bayes(X_test[i],x_y_prob,y_prob,num_all)
        pred_label.append(label)
        if(label==Y_test[i]):
            num_right+=1
        if((i+1)%200==0):
            print(num_right/(i+1))
            x_label.append(i+1)
            y_label.append(num_right/(i+1))
    x_label.append(len(X_test))
    y_label.append(num_right/len(X_test))
    #return pred_label
    return pred_label,x_label,y_label


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
    #words_list1 = words_freq_proc(words_list, 15)
    X_train, X_test, Y_train, Y_test = train_test_split(words_list, labels_list, test_size=0.2, random_state=42)

    with open('.\\tmp\\X_train.txt','w') as data:
        data.write(str(X_train))
    with open('.\\tmp\\Y_train.txt','w') as data:
        data.write(str(Y_train))
    with open('.\\tmp\\X_test.txt','w') as data:
        data.write(str(X_test))
    with open('.\\tmp\\Y_test.txt','w') as data:
        data.write(str(Y_test))

#if __name__=="__main__":
    #print("begin")
    #flag=True
    #if(flag):
    #    main()
    #else:
     #   data_save()
     #   main()


