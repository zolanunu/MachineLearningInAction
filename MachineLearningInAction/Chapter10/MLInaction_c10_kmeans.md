---
title: 第三部分 无监督学习 第十章 (机器学习实战笔记)
date: 2017-12-28
tags: [机器学习]
categories: [原创]
---

## 第十章 利用K-均值聚类算法对未标注数据分组

### 聚类

什么是聚类算法呢，相比于分类算法，很明了了。聚类算法是无监督的，也就是样本都没有任何标签，而分类对象的标签已经。将相似的对象聚成簇。簇内的对象越相似，聚类效果就越好。

下面介绍K-均值聚类的算法。K-均值：是将n个样本聚类到k个簇中，每个簇的中心采用簇中所含的均值计算得到的。

### K-均值聚类算法

优点： 容易实现

缺点：可能收敛到局部最小值，在大规模数据集上收敛较慢。

#### 伪代码

```
创建k个点作为起始质心（经常是随机选择）
当任意一个点的簇分配结果发生改变时：
	对数据集中的每个数据点
		对每个质心
			计算质心与数据点之间的距离  
		将数据点分配到距其最近的簇
	对每个簇，计算簇中所有点的均值，并将其作为簇的质心。（更新质心）
```

复杂度：时间复杂度：O（mnkd），其中 m是样本的个数，n为维数，k是迭代的次数，d为聚类中心的个数。空间复杂度：O（mn）

#### python实践

```
'''
k Means Clustering for Ch10 of Machine Learning in Action
'''
from numpy import *

def loadDataSet(fileName):
 # general function to parse tab delimited floats
    dataMat = []                
 # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) 
        # map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) 
    # la.norm(vecA-vecB)

def randCent(dataSet, k): # 在给定的数据集上构建一个包含k个随机质心的集合
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))# create centroid mat
    for j in range(n): # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids
    
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))# create mat to assign data points 
                                      # to a centroid, also holds SE of each point
    # 用于存放该样本属于哪类及质心距离
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print centroids
        for cent in range(k):# recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]# get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) # assign centroid to mean 
    return centroids, clusterAssment
```

### 二分K-均值算法

二分K-均值是为了解决简答的K-均值算法收敛于局部最小的问题。

算法思想：一开始将所有样本点作为一个簇，然后将该簇一分为二。然后选择其一进行划分，选择哪一个簇的原则是：对其划分可以最大程度降低SSE的值。

#### 伪代码

```
将所有点看成一个簇
当簇数目小于k的时候：
	对于每一个簇：
		计算总误差
		在给定的簇上面进行K-均值聚类k=2
		计算该簇一分为二之后的总误差
	选择一个总误差最小的簇继续划分
```

问题：如何进行二分呢？随机？

```
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0] # 创建一个初始的簇
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            # get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])
	    # compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
	# change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
	# replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment
```

### 实战

对地图上的点进行聚类

书中假设的情况是：Drew要进城参加朋友的生日，其他人也会过去，所以提供一个方案，参加生日的人都给出一个想去的地方，然后这些地址组成一个列表，有70个位置。保存在portland_Clubs.txt。将这些地址进行聚类，然后来安排行程。

```
import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'
    #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print yahooApi
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print "%s\t%f\t%f" % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print "error fetching"
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],\
        ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], \
    myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
```

### 补充

相关的算法改进以及应用，paper啊

待更

### 总结

聚类是一个无监督的算法，无监督是事先不知道要寻找的内容，也就是没有目标变量。聚类将数据点归到多个簇中。



