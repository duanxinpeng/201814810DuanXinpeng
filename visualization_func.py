import matplotlib.pyplot as plt
#函数式作图
#http://www.cnblogs.com/vamei/archive/2013/01/30/2879700.html#commentform
# 1D data
x = [1,2,3,4,5]
y = [2.3,3.4,1.2,6.6,7.0]
y1=[3,4,6,7,8]

plt.figure(figsize=(12,6))

plt.subplot(231)
plt.plot(x,y,x,y1)
plt.title("plot")

plt.subplot(232)
plt.scatter(x, y)
plt.title("scatter")

plt.subplot(233)
plt.pie(y)
plt.title("pie")

plt.subplot(234)
plt.bar(x, y)
plt.title("bar")

# 2D data
import numpy as np
delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z    = Y**2 + X**2

plt.subplot(235)
plt.contour(X,Y,Z)
plt.colorbar()
plt.title("contour")

# read image
import matplotlib.image as mpimg
img=mpimg.imread('marvin.jpg')

plt.subplot(236)
plt.imshow(img)
plt.title("imshow")

plt.show()