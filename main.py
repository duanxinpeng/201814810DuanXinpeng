import data_manager as dm
import math
import knn
from sklearn.model_selection import train_test_split

FREQUENCY=15



def _tf_idf(word_tf,word_doc_tf,word_df,num_doc,doc_word_tf):
    '''
    计算word_tf对应的tf-idf向量，以及其对应的tf-idf_train向量数组
    :param word_tf:
    :param word_doc_tf:
    :param word_df:
    :return:
    '''
    x_test=[]
    x_train=[]
    index_train=[]
    doc_index=set()
    for word in word_tf.keys():
        idf=math.log(num_doc/word_df.get(word))
        x_test.append((1+math.log(word_tf[word]))*idf)
        doc_index=doc_index|(word_doc_tf[word].keys())###合并所有出现的文档的index
    for index in doc_index:
        tmp=[]
        for i in range(len(word_tf.keys())):
            tf=doc_word_tf[index].get(list(word_tf.keys())[i],0)
            if(tf!=0):
                tf=1+math.log(tf)
            tmp.append(tf*math.log(num_doc/word_df.get(list(word_tf.keys())[i])))
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
    word_doc_tf, word_df,doc_word_tf=dm.words_statistics(words_list,FREQUENCY)
    X_train, X_test, Y_train, Y_test=train_test_split(doc_word_tf, labels_list, test_size=0.2, random_state=42)
    num_test=len(X_test)
    num_docs=len(doc_word_tf)
    num_right=0
    print('Prepare Finished!')
    for i in range(num_test):
        x_test,x_train,index_train=_tf_idf(X_test[i],word_doc_tf,word_df,num_docs,doc_word_tf)
        y_train=[labels_list[j] for j in index_train]
        y_eval=knn.knn_cal(x_test,x_train,y_train,5)
        if(y_eval==Y_test[i]):
            num_right=num_right+1
        index+=1
        if(index%10==0):
            print(index,'  ',num_right/i)
    return num_right/num_test

def data_save():
    '''
    读取文件，并保存这些数据
    :return:
    '''
    docs_list,labels_list=dm.doc_load()
    words_list=dm.doc_split(docs_list)
    word_doc_tf, word_df,doc_word_tf=dm.words_statistics(words_list,FREQUENCY)
    with open('.\\tmp\\word_doc_tf.txt','w') as data:
        data.write(str(word_doc_tf))
    with open('.\\tmp\\word_df.txt','w') as data:
        data.write(str(word_df))
    with open('.\\tmp\\doc_word_tf.txt','w') as data:
        data.write(str(doc_word_tf))
    with open('.\\tmp\\labels_list.txt','w') as data:
        data.write(str(labels_list))

def data_load():
    with open('.\\tmp\\word_doc_tf.txt','r') as data:
        tmp=data.read()
        word_doc_tf=eval(tmp)
    with open('.\\tmp\\word_df.txt','r') as data:
        tmp=data.read()
        word_df=eval(tmp)
    with open('.\\tmp\\doc_word_tf.txt','r') as data:
        tmp=data.read()
        doc_word_tf=eval(tmp)
    with open('.\\tmp\\labels_list.txt','r') as data:
        tmp=data.read()
        labels_list=eval(tmp)
    return word_doc_tf,word_df,doc_word_tf,labels_list

def compute_acc_without_reload():
    '''
    计算准确率
    :return: 返回准确率
    '''
    index=0
    # docs_list,labels_list=dm.doc_load()
    # words_list=dm.doc_split(docs_list)
    word_doc_tf, word_df,doc_word_tf,labels_list=data_load()
    X_train, X_test, Y_train, Y_test=train_test_split(doc_word_tf, labels_list, test_size=0.2, random_state=42)
    num_test=len(X_test)
    num_docs=len(doc_word_tf)
    num_right=0
    print('Prepare Finished!')
    for i in range(num_test):
        x_test,x_train,index_train=_tf_idf(X_test[i],word_doc_tf,word_df,num_docs,doc_word_tf)
        y_train=[labels_list[j] for j in index_train]
        y_eval=knn.knn_cal(x_test,x_train,y_train,5)
        if(y_eval==Y_test[i]):
            num_right=num_right+1
        index+=1
        if(index%10==0):
            print(index,'  ',num_right/i)
    return num_right/num_test