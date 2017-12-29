# -*- coding: utf-8 -*-

'''
Create on 2017/12/29
FP-growth Clustering for c12 of Machine Learning in Action
@author: zolanunu
'''
# FP树的数据结构

class treeNode:
	def __init__(self, namevalue, numOccur, parentNode):
		self.name = namevalue
		self.count = numOccur
		self.nodeLink = None # 用来链接相似的元素项
		self.parent = parentNode
		self.children = {} # 存放节点的子节点

	def inc(self, numOccur):# count变量变化
		self.count += numOccur

	def disp(self, ind = 1):# 将树以文本行为表现出来
		print (' ' * ind), self.name, ' ', self.count
		for child in self.children.values():
			child.disp(ind+1)

def createTree(dataSet, minSup = 1): # FP树构建函数
	headerTable = {}
	for trans in dataSet:# 第一遍遍历数据库,构建HeaderTable头指针
		for item in trans:
			headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
	for k in headerTable.keys(): # 过滤掉不满足要求的数据项
		if headerTable[k] < minSup:
			del(headerTable[k])
	freqItemSet = set(headerTable.keys())
	if len(freqItemSet) == 0:
		return None, None
	for k in headerTable:
		headerTable[k] = [headerTable[k], None]
	retTree = treeNone('Null Set', 1, None) # 头
	for transSet, count in dataSet.items(): # 第二遍遍历数据库
		localD = {}
		for item in transSet:
			localD = {}
			if item in freqItemSet:
				localD[item] = headerTable[item][0] # 拿到计数

def updateTree(items, inTree, headerTable, count):
	if items[0] in inTree.children: # 事务中的第一个元素是否作为子节点存在，存在则计数
		inTree.children[items[0]].inc(count)
	else: # 不存在创建节点
		inTree.children[items[0]] = treeNode(items[0], count, inTree)
		if headerTable[items[0]][1] == None: # 头指针更新，指向新的节点
			headerTable[items[0]][1] = inTree.children[items[0]]
		else:
			updateHeaderTable(headerTable[items[0]][1], inTree.children[items[0]])
	if len(items) > 1:
		updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
	while(nodeToTest.nodeLink != None):
		nodeToTest = nodeToTest.nodeLink
	nodeToTest.nodeLink = targetNode

def ascendTree(leafNode, prefixPath):
	if leafNode.parent != None:
		prefixPath.append(leafNode.name)
		ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)

        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #print 'condPattBases :',basePat, condPattBases
        #2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        #print 'head from conditional tree: ', myHead
        if myHead != None: #3. mine cond. FP-tree
            #print 'conditional tree for: ',newFreqSet
            #myCondTree.disp(1)            
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

import twitter
from time import sleep
import re

def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)    
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY, 
                      access_token_secret=ACCESS_TOKEN_SECRET)
    #you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1,15):
        print "fetching page %d" % i
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages

def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList

if __name__ == '__main__':
	rootNode = treeNode('pyramid', 9, None)
	rootNode.children['eye'] = treeNode('eye', 13, None)
	rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
	rootNode.disp()