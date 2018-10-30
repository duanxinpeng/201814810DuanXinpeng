
## Ansi(Ascii)--DBCS[GB2312--GBK--GB18030(少数民族汉字)]--UNICODE[UTF8-unicode的实现方式之一]
import os
from nltk.corpus import stopwords as pw
import math
from textblob import *
import numpy as np
import nltk
from nltk.corpus import wordnet
import re

import operator
from sklearn.model_selection import train_test_split



labellist=["alt.atheism","comp.graphics","comp.os.ms-windows.misc","comp.sys.ibm.pc.hardware","comp.sys.mac.hardware","comp.windows.x","misc.forsale","rec.autos","rec.motorcycles","rec.sport.baseball","rec.sport.hockey","sci.crypt","sci.electronics","sci.med","sci.space","soc.religion.christian","talk.politics.guns","talk.politics.mideast","talk.politics.misc","talk.religion.misc"]
##split bigstring to wordlist
def textParse(bigString):
    #tb=TextBlob(bigString)
    #words=tb.words.singularize().lemmatize().lower()##
    #bigString=TextBlob(bigString)
    #bigString.correct()
    bigString.lower()
    #words=nltk.wordpunct_tokenize(bigString)
    words=re.split('[^a-z]*',bigString)## 以所有除字母以外的符号作为分隔符进行分词
    words = [word for word in words if len(word) >= 3]  ##去掉长度小于三的词
    words=WordList(words).singularize()#复数变单数
    words=[word.lemmatize('v') for word in words]#过去式、进行时变一般形式
    cacheStopWords=pw.words("english")#得到stopwords
    words=[word for word in words if word not in cacheStopWords]## remove stopwords
    return words

#对dataStr进行预处理,去掉频率低于15的词语，并返回对应的vec
def preDataStr(dataStr):
    numDocs=len(dataStr)
    vocabSet=createVocabList(dataStr)
    numWords=len(vocabSet)

    #这样肯定是不行的，这样的复杂度太高了
    '''
    for word in vocabSet:
        num=0
        for doc in dataStr:
            if(word in doc):
                num=num+1
        df.append(num)
    newVocabSet=[]
    for i in range(numWords):
        if(df[i]>15):
            newVocabSet.append(vocabSet[i])
    '''

    wf={}# word frequency in all documents
    tf=[]
    df={}# times of word in all document
    for doc in dataStr:## similar to invert index
        thefirst=True# justfy whether is the first time
        tmptf={}
        for word in doc:
            if(tmptf.__contains__(word)):
                tmptf[word]=tmptf[word]+1
            else:
                tmptf[word]=1

            if(wf.__contains__(word)):
                wf[word]=wf[word]+1
                if(thefirst):
                    df[word]=df[word]+1
                    thefirst=False
            else:
                wf[word]=1
                df[word]=1
                thefirst=False
        #tf.append(wf.copy())### memory error??不能用wf,因为wf是在不断的增长的！！
        tf.append(tmptf)
    newVocabSet=[]

    for word in vocabSet:# delete the wf <=15
        if(wf[word]>15):
            newVocabSet.append(word)

    numnew=len(newVocabSet)
    dataset=np.zeros((numDocs,numnew))
    for i in range(numDocs):
        for j in range(numnew):
            normalTF=0
            if(tf[i].__contains__(newVocabSet[j])):
                normalTF=1+math.log(tf[i][newVocabSet[j]])
            else:
                normalTF=0
            dataset[i][j]=normalTF*df[vocabSet[j]]

    return dataset



##get the vector model of wordlists
'''
def getVec(datastrs):
    vocabSet=createVocabList(datastrs)
    docNum=len(datastrs)
    docNum1=10000
    numVoca=len(vocabSet)
    #tf=np.zeros((docNum,numVoca))
    #df=np.zeros(numVoca)
    idf=np.zeros(numVoca)

    dataset=np.zeros((docNum1,numVoca))
    for wordindex in range(numVoca):
    #for wordindex,word in vocabSet:
        onedf=0
        for i in range(docNum):
        #for docindex,doc in datastrs:
            #wordlist=WordList(doc)
            #tf[docindex][wordindex]=wordlist.count(word)## Compute term frequency
            if(vocabSet[wordindex] in datastrs[i]):
                onedf=onedf+1
        #df[wordindex]=onedf
        idf[wordindex]=math.log(docNum/onedf)
        for i in range(docNum1):
        #for docindex,doc in datastrs:
            #max normalization
            #thetf=tf[docindex][wordindex]/max(tf[docindex])
            ## sub-linear TF scaling
            thetf=0
            wordlist = WordList(datastrs[i])
            tf=wordlist.count(vocabSet[wordindex])
            if(tf>0):
                thetf=1+math.log(tf)
            dataset[i][wordindex]=thetf*idf[wordindex]

    np.save("dataset.npy",dataset)
    return dataset
'''
## get a vocabulary set from a dataset
def createVocabList(dataset):
    vocabSet=set([])
    for doc in dataset:
        vocabSet=vocabSet|set(doc)
    return list(vocabSet)

## load documnts from the directory,and return the vsm of all the documents
def loadData():
    dirs = os.listdir('G:\\Course\\DataMining\\201814810DuanXinpeng\\20news-18828')
    dataStrs = []
    fullWords=[]
    labls = []
    label=0    # there are 20 classes.
    for dir in dirs:
        for doc in os.listdir('G:\\Course\\DataMining\\201814810DuanXinpeng\\20news-18828\\' + dir):
            bigstring = open('G:\\Course\\DataMining\\201814810DuanXinpeng\\20news-18828\\' + dir + '\\' + doc,
                             'rb').read().decode('GBK', 'ignore')
            words=textParse(bigstring)
            dataStrs.append(words)
            fullWords.extend(words)
            labls.append(label)
        label+=1
    #np.save("dataStr.npy", dataStrs)
    #np.save("labels.npy",labls)
    dataset=preDataStr(dataStrs)
    #dataStr=np.load('dataStr.npy')
   # labls=np.load('labels.npy')
    #labls=[]
    X_train,X_test,y_train,y_test=train_test_split(dataset,labls,test_size=0.2,random_state=42)
    #accuracy=test(X_test,X_train,y_test,y_train)
    #return accuracy
    return X_train,X_test,y_train,y_test

##普通knn
def knn(inX,dataSet,labels,k):
    numDoc=len(dataSet)
    dataSet=np.array(dataSet)

    ## Euclidean distance
    inX=np.array(inX)
    diffMat=np.tile(inX,(numDoc,1))-dataSet
    sqDiffMat=diffMat**2
    sqDiffMat=sqDiffMat.sum(axis=1)
    distance=sqDiffMat**0.5


    ##cosine
    cos=[]
    for i in range(numDoc):
        cos.append(np.dot(inX,dataSet[i]))
    lenInX=inX**2
    lenInX=lenInX.sum()**0.5
    lenDataSet=dataSet**2
    lenDataSet=lenDataSet.sum(axis=1)**0.5
    lenth=lenDataSet*lenInX
    cos=cos/lenth

    sortedDistIndecies=distance.argsort()# 从小到大返回相应元素在原数组的index。
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndecies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def test(testSet,dataSet,testLabels,dataLabels):
    dataNum=len(dataSet)
    testNum=len(testSet)
    right=0
    wrong=0
    for i in range(testNum):
        res=knn(testSet[i],dataSet,dataLabels,5)
        if(res==testLabels[i]):
            right=right+1

    return right/dataNum

if __name__=="__main__":
    dataStr,labels=loadData()
    dataset=preDataStr(dataStr)