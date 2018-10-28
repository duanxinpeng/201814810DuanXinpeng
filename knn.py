
def cos_compute()
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