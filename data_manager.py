import os
import re
import textblob as tb
from nltk.corpus import stopwords as pw
#global variable
newsPath='G:\\Course\\DataMining\\201814810DuanXinpeng\\20news-18828'


def doc_load():
    '''
    :return:文档字典
    '''
    dirs = os.listdir(newsPath)#列出newPath所在文件夹中的所有文件夹的名字，返回一个文件夹名组成的数组
    docs_list=[]#创建一个空数组,存储文档
    labels_list=[]#存储文档类别
    label=0#类别
    for dir in dirs:#循环，遍历所有文件夹
        docs=os.listdir(newsPath+'\\'+dir)#文件夹的路径
        for doc in docs:#遍历文件夹中的所有文档
            with open(newsPath+'\\'+dir+ '\\' + doc,'rb') as data:#打开文档
                docs_list.append(data.read())#读出文档
            labels_list.append(label)
        label=label+1
    return docs_list,labels_list

def __string_split(str):
    '''
    :param str:
    :return: 把str分割成的单词数组
    '''
    #tb=TextBlob(bigString)
    #words=tb.words.singularize().lemmatize().lower()##
    #bigString=TextBlob(bigString)
    #bigString.correct()
    str.lower()
    #words=nltk.wordpunct_tokenize(bigString)
    words=re.split('[^a-z]*',str)## 以所有除字母以外的符号作为分隔符进行分词
    words = [word for word in words if len(word) >= 3]  ##去掉长度小于三的词
    words=tb.WordList(words).singularize()#复数变单数
    words=[word.lemmatize('v') for word in words]#过去式、进行时变一般形式
    cacheStopWords=pw.words("english")#得到stopwords
    words=[word for word in words if word not in cacheStopWords]## remove stopwords
    return words

def doc_split(docs_list):
    '''
    :param :docs_list: 文档字典
    :return words_list
    '''
    words_list=[]
    for doc in docs_list:
        words_list.append(__string_split(str(doc)))
    return words_list

def words_statistics(words_list):
    '''
    :param words_dict:
    :return:vocab:检测词频，创建词库
    '''
    index=0
    word_frequency=dict()# 单词在所有文档中出现的频率
    word_df=dict()
    word_doc_tf=dict()#倒排索引
    for doc in words_list:
        first_time = True  # 判断是否是在doc中第一次出现，用于统计df
        print(index)
        index=index+1
        for word in doc:
            word_frequency[word] = word_frequency.get(word, 0) + 1### word_frequency
            if(first_time):###word_df
                word_df[word]=word_df.get(word,0)+1
                first_time=False
            if(word_doc_tf.__contains__(word)):#word_doc_tf
                word_doc_tf[word][words_list.index(doc)]=word_doc_tf[word].get(words_list.index(doc),0)+1
            else:
                word_doc_tf[word]={words_list.index(doc):1}


    return word_doc_tf,word_frequency,word_df
