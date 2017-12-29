# -*- coding: utf-8 -*-
'''
Create on 2017/12/28
Apriori Clustering for c11 of Machine Learning in Action
@author: zolanunu
'''

from numpy import *

def loadDataSet():
	return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
	C1 = []
	for transaction in dataSet:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])
	C1.sort()
	return map(frozenset, C1)
def scanD(D, Ck, minSupport):
	ssCnt = {}
	for tid in D:
		for can in Ck:
			if can.issubset(tid):
				if not ssCnt.has_key(can):
					ssCnt[can] = 1
				else:
					ssCnt[can] += 1
	numItems = float(len(D))
	retList = []
	supportData = {}
	for key in ssCnt:
		support = ssCnt[key] / numItems
		if support >= minSupport:
			retList.insert(0, key)
		supportData[key] = support
	return retList, supportData

# 通过频繁项集Lk和项集的个数生成候选项集Ck+1 
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2:
            	# 如果前k-2项相同时候，将两个集合合并
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


'''
L: 频繁项集列表
supportData： 包含哪些频繁项集支持数据的字典
minconf： 最小可信度阈值
'''
def generateRules(L, supportData, minConf=0.7):  # 频繁项集挖掘关联规则
    bigRuleList = [] # 保存包含可信度的规则列表
    for i in range(1, len(L)):# 只保留两个或者两个元素以上的集合
        for freqSet in L[i]: # 遍历某个频繁项集的元素
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else: # 计算规则的可信度，并过滤出满足最小可信度要求的规则
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def generateRules2(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                # 三个及以上元素的集合
                H1 = calcConf(freqSet, H1, supportData, bigRuleList, minConf)
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                # 两个元素的集合
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def rulesFromConseq31(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    while (len(freqSet) > m): # 判断长度 > m，这时即可求H的可信度
        H = calcConf(freqSet, H, supportData, brl, minConf)
        if (len(H) > 1): # 判断求完可信度后是否还有可信度大于阈值的项用来生成下一层H
            H = aprioriGen(H, m + 1)
            m += 1
        else: # 不能继续生成下一层候选关联规则，提前退出循环
            break

def generateRules3(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            rulesFromConseq2(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList
 
def rulesFromConseq2(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > m): # 判断长度改为 > m，这时即可以求H的可信度
        Hmpl = calcConf(freqSet, H, supportData, brl, minConf)
        if (len(Hmpl) > 1): # 判断求完可信度后是否还有可信度大于阈值的项用来生成下一层H
            Hmpl = aprioriGen(Hmpl, m + 1)
            rulesFromConseq2(freqSet, Hmpl, supportData, brl, minConf) # 递归计算，不变
'''
calConf: 计算规则的可信度，并筛选出满足最小可信度要求的规则
freqSet: 频繁项集
H: 候选规则集合
supportData: 保存项集支持度
brl: 保存生成的规则
minConf: 最小置信度
'''
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
	# 计算规则的可信度，并过滤出满足最小可信度要求的规则
    prunedH = [] # 保存规则列表的右部
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf: 
            print freqSet-conseq,'-->',conseq,'conf:',conf
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

'''
根据当前的候选规则集合H生成下一层候选规则集
freqSet: 频繁项集
H: 候选规则集合
supportData: 保存项集支持度
brl: 保存生成的规则
minConf: 最小置信度
'''
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print itemMeaning[item]
        print "           -------->"
        for item in ruleTup[1]:
            print itemMeaning[item]
        print "confidence: %f" % ruleTup[2]
        print       #print a blank line

from time import sleep
from votesmart import votesmart

votesmart.apikey = 'get your api key first'

def getActionIds():
	# 保存actionId和标题
	actionIdList = []; billTitleList = []
	fr = open('recent20bills.txt')
	for line in fr.readlines():
		billNum = int(line.split('\t')[0])
		try:
			billDetail = votesmart.votes.getBill(billNum) # 获得billDetail的对象
			# 遍历议案中的所有行为，寻找有投票行为的数据
			for action in billDetail.actions:
				if action.level == 'House' and \
				(action.stage == 'Passage' or action.stage == 'Amendment Vote'):
					actionId = (int)(action.actionId)
					print('bill: %d has actionId: %d' %(billNum, actionId))
					actionIdList.append(actionId)
					billTitleList.append(line.strip().split('\t')[1])
		except:
			print('problem getting bill %d' % billNum)
		sleep(1)
	return actionIdList, billTitleList
'''
基于投票数据的事务列表填充函数
创建事务数据库
'''
def getTransList(actionIdList, billTitleList):
    itemMeaning = ['Republican', 'Democratic']
    for billTitle in billTitleList:
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print 'getting votes for actionId: %d' % actionId
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print "problem getting actionId: %d" % actionId
        voteCount += 2
    return transDict, itemMeaning

if __name__ == "__main__":
	data = loadDataSet()
	# print(data)
	C1 = createC1(data)
	# print C1
	D = map(set, data)
	# print(D)
	L1, supportDataSet = scanD(D, C1, 0.5)
	# print(L1)
	# print(supportDataSet)
	L, d = apriori(data)
	# print(L)
	# print(d)
	acid, billtitleid = getActionIds()
	#print(acid)
	#print(billtitleid)
	transDict, item = getTransList(acid, billtitleid)
	dataset = [transDict[key] for key in transDict.keys]