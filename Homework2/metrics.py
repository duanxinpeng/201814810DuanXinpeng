import Homework2.naive_bayes as nb
import Homework2.naive_bayes_multipoly as nbm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import numpy as np



def plotAcc():
    # x1=[1,2,3,4,5]
    # x2=[1,2,3,4,5]
    # y1=[1,2,3,4,5]
    # y2=[2,3,4,5,6]
    pred1,x1,y1=nb.main()
    pred2,x2,y2=nbm.main()
    with open('x1.txt', 'w') as data:
        data.write(str(x1))
    with open('y1.txt', 'w') as data:
        data.write(str(y1))
    with open('x2.txt', 'w') as data:
        data.write(str(x2))
    with open('y2.txt', 'w') as data:
        data.write(str(y2))
    #ax=plt.subplot(1,1,1)
    p1=plt.scatter(x1,y1)
    p2=plt.scatter(x2,y2)
    plt.legend(handles = [p1, p2,], labels = ['Bernoulli', 'Polynomial'], loc = 'best')
    plt.title("acc-testNum")
    plt.show()
def metricEvaluate():
    # pred_label1,x1,y1=nb.main()
    # pred_label2,x2,y2=nbm.main()
    # with open('pred_Bernoulli.txt', 'w') as data:
    #    data.write(str(pred_label1))
    # 利用sklearn中的classification_report 可以得到的结果
    # with open('pred_polynomial.txt', 'w') as data:
    #     data.write(str(pred_label2))
    with open('.\\tmp\\Y_test.txt', 'r') as data:
        tmp = data.read()
        Y_test = eval(tmp)
    with open('.\\tmp\\pred_bernoulli.txt', 'r') as data:
        tmp = data.read()
        pred_bernoulli = eval(tmp)
    with open('.\\tmp\\pred_polynomial.txt', 'r') as data:
        tmp = data.read()
        pred_polynomial = eval(tmp)
    Y_name = os.listdir('20news-18828')
    classification_report(np.array(Y_test), np.array(pred_bernoulli), target_names=Y_name)
    classification_report(np.array(Y_test), np.array(pred_polynomial), target_names=Y_name)
    accuracy_score(Y_test, pred_bernoulli)
    accuracy_score(Y_test, pred_polynomial)
if __name__ == '__main__':
    #metricEvaluate()
    plotAcc()
