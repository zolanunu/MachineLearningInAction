# -*- coding: utf-8 -*-
'''
Create on 2017/12/28
k Means Clustering for c10 of Machine Learning in Action
@author: zolanunu
'''

from numpy import *

def loadDataSet(fileName):
	dataMat = []
	# 将文本文件导入一个列表，每一行以tab分隔符的浮点数
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float, curLine)
		dataMat.append(fltLine) # 每一列都添加到dataMat
	return dataMat

def distEclud(vecA, vecB):
	# 计算两个向量的欧氏距离
	return sqrt(sum(power(vecA-vecB, 2)))

def randCent(dataSet, k):
	# 在给定的数据集上随机构建一个包含k个簇的集合
	# 随机质心必须是在数据集的边界以内的
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids


# K-均值算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2))) # 存放某个样本属于那个质心，以及距离质心的距离
    centroids = createCent(dataSet, k)
    #print("initial centroids:")
    #print(centroids)
    #print("training k ")
    #count = 0
    clusterChanged = True
    while clusterChanged: # 直到收敛
        clusterChanged = False
        #count = count + 1
        #print(count)
        for i in range(m):# 对每一个样本进行质心距离分析，然后进行将其分类
            minDist = inf; minIndex = -1
            for j in range(k): # 计算某个样本和质心的距离
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2 # 将样本分到距离最近的簇中
        #print centroids
        for cent in range(k): # 对所有的样本都归簇好，然后重新选择质心
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

# 二分K-均值算法

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0] # 开始的质心是所有样本的均值
    centList =[centroid0]
    for j in range(m): # 将所有的样本都归为一个簇中，质心距离计算
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k): # 当簇的数目达到k的时候，停止划分
        lowestSSE = inf # SSE：Sum of Square Error 误差平方和
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE: # 不划分和划分的误差之和和最小总误差比较
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    return mat(centList), clusterAssment

import urllib
import json
# 书中给出的例子是说雅虎网站上给出了一个免费的地址转换API，该API对给定的地址返回该地址对应的纬度和经度
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    # 将返回类型为jsom
    params['flags'] = 'J'
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params) # 将创建的字典可以通过URL进行传递的字符串格式
    yahooApi = apiStem + url_params
    print yahooApi # url
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read()) # 将读取的数据以json格式返回

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        # 该函数打开一个以tab分割的文本文件，获取其中2，3列的结果，在输入到上面的函数中
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
# 对地理位置进行聚类
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
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

if __name__ == "__main__":
    # print('load')
    data = mat(loadDataSet('./testSet.txt'))
	#print(data)
    myCen, clusAss = kMeans(data, 4)
    clusterClubs(5)