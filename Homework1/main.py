import math
from Homework1 import knn, data_manager as dm
from Homework1 import knn
from sklearn.model_selection import train_test_split
import numpy as np

FREQUENCY=15



def _tf_idf(word_tf_test,word_doc_tf_train,word_idf_train,num_doc_train,doc_word_tf_train):
    '''
    计算word_tf对应的tf-idf向量，以及其对应的tf-idf_train向量数组
    :param word_tf:
    :param word_doc_tf:
    :param word_df:
    :return:
    '''

    word_in_test=[word for word in word_tf_test if word in list(word_doc_tf_train.keys())]#去掉test文档中没有在训练数据集中出现过的单词
    x_test=[]
    x_train=[]
    index_train=[]
    doc_index=set()
    for word in word_in_test:
        #idf=math.log(num_doc_train/word_df_train.get(word))#其实都是根据训练数据集来进行计算的，可以先计算清楚。
        x_test.append((1+math.log(word_tf_test[word]))*word_idf_train[word])
        doc_index=doc_index|(word_doc_tf_train[word].keys())###合并所有在训练数据集中出现过的文档的index
    for index in doc_index:
        tmp=[]
        for word in word_in_test:
            tf=doc_word_tf_train[index].get(word,0)
            if(tf!=0):
                tf=1+math.log(tf)
            tmp.append(tf*word_idf_train[word])
        x_train.append(tmp)
        index_train.append(index)
    return x_test,x_train,index_train




def compute_acc():
    '''
    计算准确率
    :return: 返回准确率
    '''
    index=0
    docs_list,labels_list=dm.doc_load()
    words_list=dm.doc_split(docs_list)
    words_list=dm.words_freq_proc(words_list,FREQUENCY)
    X_train, X_test, Y_train, Y_test = train_test_split(words_list, labels_list, test_size=0.2, random_state=42)
    word_doc_tf_train, word_idf_train,doc_word_tf_train=dm.words_statistics(X_train)
    word_doc_tf_test,word_idf_test,doc_word_tf_test=dm.words_statistics(X_test)

    num_test=len(doc_word_tf_test)
    num_docs_train=len(doc_word_tf_train)
    num_right=0
    print('Prepare Finished!')
    for i in range(num_test):
        x_test,x_train,index_train=_tf_idf(doc_word_tf_test[i],word_doc_tf_train,word_idf_train,num_docs_train,doc_word_tf_train)
        y_train=[Y_train[j] for j in index_train]
        y_eval= knn.knn_cal(x_test, x_train, y_train, 5)
        if(y_eval==Y_test[i]):
            num_right=num_right+1
        index+=1
        if(index%10==0):
            print(index,'  ',num_right/i)
    return num_right/num_test


##以下待修改
def data_save():
    '''
    读取文件，并保存这些数据
    :return:
    '''
    docs_list,labels_list=dm.doc_load()
    words_list=dm.doc_split(docs_list)
    words_list=dm.words_freq_proc(words_list,FREQUENCY)
    X_train, X_test, Y_train, Y_test = train_test_split(words_list, labels_list, test_size=0.2, random_state=42)
    word_doc_tf_train, word_idf_train,doc_word_tf_train=dm.words_statistics(X_train)
    word_doc_tf_test,word_idf_test,doc_word_tf_test=dm.words_statistics(X_test)

    with open('.\\tmp\\word_doc_tf_train.txt','w') as data:
        data.write(str(word_doc_tf_train))
    with open('.\\tmp\\word_idf_train.txt','w') as data:
        data.write(str(word_idf_train))
    with open('.\\tmp\\doc_word_tf_train.txt','w') as data:
        data.write(str(doc_word_tf_train))
    with open('.\\tmp\\doc_word_tf_test.txt','w') as data:
        data.write(str(doc_word_tf_test))
    with open('.\\tmp\\Y_train.txt','w') as data:
        data.write(str(Y_train))
    with open('.\\tmp\\Y_test.txt','w') as data:
        data.write(str(Y_test))

def data_load():
    with open('.\\tmp\\word_doc_tf_train.txt','r') as data:
        tmp=data.read()
        word_doc_tf_train=eval(tmp)
    with open('.\\tmp\\word_idf_train.txt','r') as data:
        tmp=data.read()
        word_idf_train=eval(tmp)
    with open('.\\tmp\\doc_word_tf_train.txt','r') as data:
        tmp=data.read()
        doc_word_tf_train=eval(tmp)
    with open('.\\tmp\\doc_word_tf_test.txt','r') as data:
        tmp=data.read()
        doc_word_tf_test=eval(tmp)
    with open('.\\tmp\\Y_train.txt','r') as data:
        tmp=data.read()
        Y_train=eval(tmp)
    with open('.\\tmp\\Y_test.txt','r') as data:
        tmp=data.read()
        Y_test=eval(tmp)
    return word_doc_tf_train,word_idf_train,doc_word_tf_train,doc_word_tf_test,Y_train,Y_test


def compute_acc_without_reload():
    index = 0
    word_doc_tf_train, word_idf_train, doc_word_tf_train, doc_word_tf_test, Y_train, Y_test = data_load()

    num_test = len(doc_word_tf_test)
    num_docs_train = len(doc_word_tf_train)
    num_right = 0
    print('Prepare already Finished!')
    for i in range(num_test):
        x_test, x_train, index_train = _tf_idf(doc_word_tf_test[i], word_doc_tf_train, word_idf_train, num_docs_train,
                                              doc_word_tf_train)
        y_train = [Y_train[j] for j in index_train]
        y_eval = knn.knn_cal(x_test, x_train, y_train, 5)
        if (y_eval == Y_test[i]):
            num_right = num_right + 1
        index += 1
        if (index % 10 == 0):
            print(index, ' ', num_right / i)
    return num_right / num_test

def compute_without_reload():
    '''
    计算准确率
    :return: 返回准确率
    '''
    index=0
    # docs_list,labels_list=dm.doc_load()
    # words_list=dm.doc_split(docs_list)
    word_doc_tf_train, word_idf_train,doc_word_tf_train,doc_word_tf_test,Y_train,Y_test=data_load()

    num_test=len(doc_word_tf_test)
    num_docs_train=len(doc_word_tf_train)
    num_right=0
    print('Prepare Finished!')
    for i in range(num_test):
        x_test,x_train,index_train=_tf_idf(doc_word_tf_test[i],word_doc_tf_train,word_idf_train,num_docs_train,doc_word_tf_train)
        y_train=[Y_train[j] for j in index_train]
        # print(x_test,"\n")
        # print(x_train,"\n")
        #print(knn.euclidean_compute(np.array(x_test),np.array(x_train)))
        print(x_test,"\n")
        # y_eval= knn.knn_cal(x_test, x_train, y_train, 25)
        # if(y_eval==Y_test[i]):
        #     num_right=num_right+1
        # index+=1
        # if(index%10==0):
        #     print(index,'  ',num_right/i)
    return num_right/num_test

if __name__ =='__main__':
    #compute_without_reload()
    compute_acc_without_reload()

