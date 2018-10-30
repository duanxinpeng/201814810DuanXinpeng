DataMining
======
Homework1 
------
共包含四个文件，vsm是一个最开始的错误思路示范，所以只有main，knn，data_manager三个文件是有用文件。
其中：
* data_manager用于数据预处理；
* knn是K-近邻的具体实现；
* main用于将两者结合在一起。

可以通过main.compute_acc()，从数据处理开始，直接得到knn准确率；<br>
也可以先通过main.data_save(),将数据处理结果保存到tmp下的临时文件中，再通过main.compute_acc_without_reload()得到knn算法的准确率。<br>
最终的准确率为:`0.79129`