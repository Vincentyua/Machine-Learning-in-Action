"""
Create by yuanhao at 2018-3-8
Happy Women's Day
github:https://github.com/Vincentyua
"""
import numpy as np
import operator
import time



def classify0(inX,dataSet,labels,k):
    """
    knn算法分类器
    :param inX: 输入的需分类的向量
    :param dataSet: 训练数据集
    :param labels: 标签向量
    :param k: 邻居数
    :return:分类结果
    """
    dataSetSize = dataSet.shape[0]
    # tile函数的作用是重复某个数组，tile(a，(b,c))重复a数组为b行c列
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    # axis=0按列相加，axis=1按行相加，无参数全部相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 按欧式距离计算
    distance = sqDistances**0.5
    # argsort函数返回的是数组值从小到大的索引值
    sortedDistIndices = distance.argsort()
    # 定义一个记录类别次数的字典
    classCount={}
    # 取出前k个元素的类别
    for i in range(k):
        # 取出类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 取出一个类别后，字典中这个类别次数加1；计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 字典(Dictionary) items() 函数以列表返回可遍历的(键, 值) 元组数组
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # sortedClassCount是按字典值降序排好的列表，返回第一个类别
    return sortedClassCount[0][0]



def file2matrix(filename):
    """
    从文本文件解析数据
    :param filename:文件名称
    :return:数据矩阵，label向量
    """
    fr = open(filename)
    # 读取文件所有内容
    arrayOfLines = fr.readlines()
    # 获得文件行数
    numberOfLines = len(arrayOfLines)
    # 创建返回的矩阵,numberOfLines行，3列
    returnMat = np.zeros((numberOfLines,3))
    # 创建类别标签向量
    classLableVector = []
    # 行的索引值
    index = 0
    for line in arrayOfLines:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片
        listFromLine = line.split('\t')
        # 将每行的前3列提取出来，放入特征矩阵对应行中
        returnMat[index,:] = listFromLine[0:3]
        # 将最后一列元素放入类别标签向量
        classLableVector.append(int(listFromLine[-1]))
        # 行索引值加1
        index += 1
    return returnMat,classLableVector


def autoNorm(dataSet):
    """
    归一化数值
    :param dataSet:原始数据矩阵
    :return:归一化矩阵，原始数据范围，原始数据最小值
    """
    # 获得数据的最值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 原始数据的范围，一会做分母
    ranges = maxVals - minVals
    # 创建新的矩阵，大小与原来一样
    normDataSet = np.zeros(np.shape(dataSet))
    # m是原始矩阵的行数
    m = dataSet.shape[0]
    # 整体操作
    normDataSet = dataSet -np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def classifyPerson():
    # 构建结果列表
    resultList = ['不喜欢','魅力一般','极具魅力']
    flyMails = float(input("每年获取的飞行常客里程数？"))
    percentTats = float(input("玩视频游戏所耗时间百分比？"))
    iceCream = float(input("每周消费的冰激凌公升数？"))
    # 加载训练集
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # 归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([flyMails,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('你可能觉得这个人：',resultList[classifierResult-1])




if __name__ == '__main__':
    start = time.time()
    classifyPerson()
    end = time.time()
    print('<运行时间为：%fs>' % (float(end - start)))
