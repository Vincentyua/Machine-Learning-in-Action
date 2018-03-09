"""
Create by yuanhao at 2018-3-8
Happy Women's Day
github:https://github.com/Vincentyua
"""
import numpy as np
import operator
import time
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as kNN


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



def img2vector(filename):
    """
    字符图像转化为向量
    :param filename:文件名称
    :return:转化后的向量
    """
    # 提前创建返回的向量1*1024
    returnVector = np.zeros((1,1024))
    fr = open(filename)
    # 循环读出每行
    for i in range(32):
        # 每行转化为列表
        lineStr = fr.readline()
        # 循环读入每行前32个元素
        for j in range(32):
            # 依次将每个元素放入返回向量
            returnVector[0,i*32+j] = int(lineStr[j])
    return returnVector

def handwritingClassTest():
    """
    手写数字识别测试算法
    """
    # 创建类别标签
    hwLabels = []
    # 将给定训练集目录的文件名加载到列表里
    trainingFileList = listdir('digits/trainingDigits')
    # 计算文件个数
    m = len(trainingFileList)
    # 创建空的训练矩阵
    trainingMat = np.zeros((m,1024))
    # 将目录下的每个文件转化为向量保存在训练矩阵中
    for i in range(m):
        # 从列表中取出文件名
        fileNameStr = trainingFileList[i]
        # 对文件名处理，切割
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        # 将从文件名提取出的类别保存在类别向量里
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s'%(fileNameStr))
    # 构建kNN分类器
    knn_sk = kNN(n_neighbors = 3, algorithm = 'auto')
    # 训练模型
    knn_sk.fit(trainingMat,hwLabels)
    # 加载测试集目录
    testFileList = listdir('digits/testDigits')
    mTest = len(testFileList)
    errorCount = 0.0
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        # 导出，测试向量
        vectorUnderTest = img2vector('digits/testDigits/%s'%(fileNameStr))
        # 得到分类结果
        classifilerResult = knn_sk.predict(vectorUnderTest)
        print('分类结果为：%d，真实结果为：%d'%(classifilerResult,classNumStr))
        if(classifilerResult != classNumStr):
            errorCount += 1
    print('\n总的错误个数为：%d'%errorCount)
    print('\n总的错误率为：%f'%(errorCount/float(mTest)))


if __name__ == '__main__':
    start = time.time()
    handwritingClassTest()
    end = time.time()
    print('<运行时间为：%fs>'%(float(end-start)))
