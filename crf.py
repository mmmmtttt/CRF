import numpy as np
import torch
import os
from dataLoader import loadData,loadLabel,loadTemplate

TEMPLATEPATH = "./train_dataset/dataset1/template.utf8"
LABELPATH = "./train_dataset/dataset1/labels.utf8"
DATAPATH = "./train_dataset/dataset1/train.utf8"
# TEMPLATEPATH = "./train_dataset/dataset2/template.utf8"
# LABELPATH = "./train_dataset/dataset2/labels.utf8"
# DATAPATH = "./train_dataset/dataset2/train.utf8"
CHECKPOINTPATH = "crf_1.pkl"

class CRF:
    def __init__(self) -> None:
        self.featureMap = dict() #所有模版规定的特征函数的输入到输出的映射，即模型训练的参数
        self.averageFeatureMap = dict() #使用平均的参数作为最终的参数结果
        self.templates = loadTemplate(TEMPLATEPATH)
        self.labels = loadLabel(LABELPATH)

    def saveModel(self,path):
        checkpoint = {'featureMap': self.featureMap,
              'averageFeatureMap':self.averageFeatureMap,
              'templates':self.templates,
              'labels': self.labels}
        torch.save(checkpoint, path)
        print("finish saving model")

    def loadModel(self,path):
        if os.path.exists(path):
            print("part1:load model from %s"%path)
            checkpoint = torch.load(path)
            self.featureMap = checkpoint['featureMap']
            self.labels = checkpoint['labels']
            self.averageFeatureMap = checkpoint['averageFeatureMap']
            self.templates = checkpoint['templates']
        else:
            print("start training new model")
    
    def getMaxScoredSequence(self,sentence,featureMap):
        """
        在当前模型参数下，viterbi算法得到一个句子的最可能的tag序列
        """
        labelNum = len(self.labels)
        length = len(sentence)
        maxScores = np.zeros([labelNum,length])
        indexFrom = np.zeros([labelNum,length]) 
        for col in range(length): #对句子中的每个位置
            for row in range(labelNum): #对当前位置可能的每个标签
                uniscore = self.getScoreFromTemplate(sentence,col,self.labels[row],'u',featureMap) #计算unigram得分
                #对每一种可能的前一个位置的状态转移 计算unigram和bigram总得分
                possibleScores = ([uniscore + maxScores[tag,col-1] +
                 self.getScoreFromTemplate(sentence,col,self.labels[tag]+"_"+self.labels[row],'b',featureMap) 
                 for tag in range(labelNum)])
                maxScores[row,col] = max(possibleScores)
                indexFrom[row,col] = possibleScores.index(maxScores[row,col])
        #解码最大可能性的tag序列
        tagResult = []
        maxScore = np.max(maxScores[:,-1])
        maxIndex = np.where(maxScores[:,-1]==maxScore)[0][0]
        tagResult.append(maxIndex)
        for i in reversed(range(1,length)):
            tagResult.insert(0,int(indexFrom[tagResult[0],i]))
        return self.labels[tagResult]

    def findFeatureKey(self,sentence,pos,tagTransfer,template):
        """
        根据模版在具体句子中扩展成具体的某个特征函数
        """
        length = len(sentence)
        characters = [sentence[pos+offset] if pos+offset>=0 and pos+offset<length else " " for offset in template]# 得到模版规定的其他位置上的字
        join = np.transpose((template, characters)).flatten() #把两个列表交错合并
        # 模版规定tag
        featureKey = "_".join(join)+"_"+tagTransfer
        return featureKey

    def getScoreFromTemplate(self,sentence,pos,tagTransfer,mode,featureMap):
        """
        计算句中某个位置在所有模版上的得分
        pos:当前位置
        mode:b-bigram ;u-unigram
        tagTransfer:如果是bigram,是 前一tag_当前tag;如果是unigram,是当前tag
        """
        score = 0
        for template in self.templates[mode]:
            feature = self.findFeatureKey(sentence,pos,tagTransfer,template)
            #查找对应的特征函数打分,不存在返回0
            score += featureMap.get(feature,0) 
        return score
    
    def updateFunctionWeight(self,sentence,pos,tagTransfer,mode,change):
        """
        更新featureMap句中某个位置在所有模版上的得分;更新averageFeatureMap计算平均参数
        change: 增加1或者增加-1
        """
        for template in self.templates[mode]:
            feature = self.findFeatureKey(sentence,pos,tagTransfer,template)
            #改变对应的特征函数的权重,不存在就加入map
            self.featureMap[feature] = self.featureMap.get(feature,0) + change
            self.averageFeatureMap[feature] = self.averageFeatureMap.get(feature,0) + change
        
    def perceptronAlgo(self,sentence,goldTag,predictTag):
        """
        对比goldTag用perceptron算法更新参数
        对每个不同的tag的位置更新对应特征函数的权重，错误的减少，正确的增加
        """
        wrongNum = 0
        #让averageFeatureMap加上这一轮的featureMap
        self.updateAverageFeatureMap()
        #修改featureMap参数
        for i,tag in enumerate(predictTag):
            if tag == goldTag[i]:
                continue
            wrongNum = wrongNum+1
            prevGoldTag = goldTag[i-1] if i>0 else " "
            prevPredTag = predictTag[i-1] if i>0 else " "
            #改变unigram
            self.updateFunctionWeight(sentence,i,goldTag[i],'u',1)#增加正确的tag的权重
            self.updateFunctionWeight(sentence,i,predictTag[i],'u',-1)#减少错误的tag的权重
            #改变bigram
            self.updateFunctionWeight(sentence,i,prevGoldTag+"_"+goldTag[i],'b',1)#增加正确的tag的权重
            self.updateFunctionWeight(sentence,i,prevPredTag+"_"+predictTag[i],'b',-1)#减少错误的tag的权重
        return wrongNum

    def updateAverageFeatureMap(self):
        """
        让averageFeatureMap先加上没有改变的featureMap的参数，之后和featureMap一起修改参数
        """
        for k,v in self.featureMap.items():
            if k in self.averageFeatureMap.keys():
                self.averageFeatureMap[k]+= v
            else:
                self.averageFeatureMap[k] = v

    def train(self,path,epoch):
        """
        对path路径下的train.utf8训练epoch轮
        """
        sentences,tags,_ = loadData(path)
        sentenceNum = len(sentences)
        print("load %s sentences"%sentenceNum)
        
        for t in range(epoch):
            totalWrongNum = 0
            totalNum = 0
            print("-------------epoch %d-------------"%t)
            for i,(sentence,goldTag) in enumerate(zip(sentences,tags)):
                predictTag = self.getMaxScoredSequence(sentence,self.featureMap)
                # print("%s\n%s"%(sentence,predictTag))
                wrongNum = self.perceptronAlgo(sentence,goldTag,predictTag)
                print("epoch %d :(%s/%s) wrong number: %s "%(t,i,sentenceNum,wrongNum))
                totalWrongNum += wrongNum
                totalNum += len(sentence)
            print("epoch %d: accuracy %s"%(t,1-totalWrongNum/totalNum))
        self.saveModel(CHECKPOINTPATH)

    def predict(self,sentences):
        result = []
        for sentence in sentences:
            result.append(''.join(self.getMaxScoredSequence(sentence,self.averageFeatureMap)))
            print(result)
        return result

if __name__=='__main__':  
    crf = CRF()
    crf.loadModel(CHECKPOINTPATH)
    # crf.train(DATAPATH,10)