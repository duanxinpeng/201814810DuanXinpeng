DataMining
======
Homework1 
------
* data_manager 加载数据；
* knn K-近邻的实现；
* main 计算并返回准确率。

可以通过main.compute_acc()，从数据处理开始，直接得到knn准确率；<br>
也可以先通过main.data_save(),将数据处理结果保存到tmp下的临时文件中，再通过main.compute_acc_without_reload()得到knn算法的准确率。<br>
最终的准确率为:`0.79129`

Homework2
------
* data_manager 加载数据
* naive_bayes 伯努利模式的实现
* naive_bayes_multipoly 多项式模式的实现
* metrics 模型评估

伯努利模式准确率：`0.7697822623473181`  
多项式模式准确率：`0.8810408921933085`  