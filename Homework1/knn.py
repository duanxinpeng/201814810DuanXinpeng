import numpy as np
import operator

def cos_compute(x_test,x_train):
    '''
    计算x_test和x_train的cos，用于比较两者是否相似，越接近1越相似！
    :param x_test:numpy.array
    :param x_train: numpy.array
    :return:
    '''
    len_test=(x_test**2).sum()**0.5
    len_train=(x_train**2).sum(axis=1)**0.5
    num_train=len(x_train)
    cos=[]
    for i in range(num_train):
        cos.append(np.dot(x_test,x_train[i]))
    cos=cos/(len_test*len_train)
    return cos

def euclidean_compute(x_test,x_train):
    '''
    计算欧几里得距离，同样用于比较相似，其值越小越相似
    :param x_test: numpy.array
    :param x_train: numpy.array
    :return:
    '''
    num_train=len(x_train)
    x_test=np.tile(x_test,(num_train,1))
    test_minus_train=x_test-x_train
    square=test_minus_train**2
    sum=square.sum(axis=1)
    distance=sum**0.5
    return distance


def knn_cal(x_test,x_train,y_train,k):
    '''
    knn方法！
    :param x_test: 测试数据
    :param x_train: 训练数据
    :param y_train: 训练数据标签
    :param k:
    :return: 返回x_test属于哪一个类
    '''
    distance=euclidean_compute(x_test,x_train)#计算x_test和x_train中的每一个元素的距离
    sortedDistIndecies=distance.argsort()#对上述计算结果进行排序
    classCount={}
    #判断在x_train中和x_test相似的前k个元素中，属于哪个类的元素更多
    for i in range(k):
        voteIlabel=y_train[sortedDistIndecies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]