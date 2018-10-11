
## Ansi(Ascii)--DBCS[GB2312--GBK--GB18030(少数民族汉字)]--UNICODE[UTF8-unicode的实现方式之一]
import os
from textblob import *
import numpy as np
##split bigstring to wordlist
def textParse(bigString):
    tb=TextBlob(bigString)
    words=tb.words.singularize()
    return words
##get the vector model of wordlists
def getVec(datastrs):
    vocabSet=createVocabList(datastrs)
    docNum=len(datastrs)
    numVoca=len(vocabSet)
    tf=np.zeros((docNum,numVoca))
    df=np.zeros(numVoca)
    for word,wordindex in vocabSet:
        for doc,docindex in datastrs:
            wordlist=WordList(doc)
            tf[docindex][wordindex]=wordlist.count(word)

    dataset=[]
    return dataset
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
    label=0
    for dir in dirs:
        for doc in os.listdir('G:\\Course\\DataMining\\201814810DuanXinpeng\\20news-18828\\' + dir):
            bigstring = open('G:\\Course\\DataMining\\201814810DuanXinpeng\\20news-18828\\' + dir + '\\' + doc,
                             'rb').read().decode('GBK', 'ignore')
            words=textParse(bigstring)
            dataStrs.append(words)
            fullWords.extend(words)
            labls.append(label)
        label+=1
    dataset=getVec(dataStrs)
    return dataset, labls

