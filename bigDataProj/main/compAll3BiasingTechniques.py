'''
Created on 19-Dec-2017

@author: Ashwin
'''
from ulisf import densratio
from sklearn import svm
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession
import numpy as np
from kliep.kliep import DensityRatioEstimator
import kmm.kmm
import random 
import matplotlib.pyplot as plt
from numpy import linspace
from constants import constants


def split(r):
    rArr = r.split(",")
    if len(rArr) == 0:
        return r
    l = []
    for x in rArr:
        l.append(x)
    return l

def readFile(sc, filePath):
    inputFile = sc.textFile(filePath)
    inputFileList = inputFile.map(split).collect()
    if np.array(inputFileList).ndim == 1:
        return np.array(inputFileList).astype(np.float)
    else:
        return np.array(inputFileList)[:, :].astype(np.float)
    
def randomDataSample(data, label, modelSize=9395):
    sampleData = []
    sampleLabel = []
    if len(data) <= modelSize:
        sampleData = data
        sampleLabel = label
    else:
        seen = []
        while len(seen) < modelSize:
            r = random.randint(0, len(data) - 1)
            if r not in seen:
                seen.append(r)
                sampleData.append(data[r])
                sampleLabel.append(label[r])
    
    sampleData = np.array(sampleData)[:, :].astype(np.float)
    sampleLabel = np.array(sampleLabel).astype(np.float) 
    
    return sampleData, sampleLabel

def getAccuracy(trainData, trainLabel, testData, testLabel, weights):
    maxacc = 0
    maxC = 0
    maxG = 0
    for c in [1, 10, 100, 1000]:
        for gamma in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
            csf2 = svm.SVC(C=c, kernel='rbf', gamma=gamma, probability=False)
            if weights.all() == 0:
                predicted = csf2.fit(trainData, trainLabel).predict(testData)
            else:
                predicted = csf2.fit(trainData, trainLabel, sample_weight=weights).predict(testData)    
            acc = accuracy_score(testLabel, predicted)
            if maxacc < acc:
                maxacc = acc
                maxC = c
                maxG = gamma
    return maxacc, maxC, maxG

def plot(noBiasacc, kmmacc, kliepacc, ulsifacc):
    if constants.trainingSet == "NYTAtrocity":
        t = linspace(500, 3000, 6)
    else:
        t = linspace(1000, 9000, 9)
    figureName = "Comparison all 3 bias technique - Train - " + constants.trainingSet +  " Test - " + constants.testSet
    plt.figure(figureName)
    plt.xlabel("Training data")
    plt.ylabel("Accuracy")
    plt.plot(t, noBiasacc, label="noBias")
    plt.plot(t, kmmacc, label="kmm") 
    plt.plot(t, kliepacc, label="pyklieb") 
    plt.plot(t, ulsifacc, label="ulsif")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
    plt.show()
      

def main():
    spark = SparkSession.builder.getOrCreate()
   
    sc = spark.sparkContext
     
    trainData = readFile(sc, constants.trainDataPath)
    trainLabel = readFile(sc, constants.trainlabelPath)
    testData = readFile(sc, constants.testDataPath)
    testLabel = readFile(sc, constants.testlabelPath)
     
    kmmacc = []
    noBiasacc = []
    ulsifacc = []
    kliepacc = []
     
    if constants.trainingSet == "NYTAtrocity":
        modelRange = [500, 1000, 1500, 2000, 2500, 2996]
    else:
        modelRange = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9395]
         
    for modelSize in modelRange:
         
        sampleTestX, sampleTestY = randomDataSample(testData, testLabel)
        sampleTrainX, sampleTrainY = randomDataSample(trainData, trainLabel, modelSize)
         
        # # without biasing techinique
        maxacc, maxC, maxG = getAccuracy(sampleTrainX, sampleTrainY, sampleTestX, sampleTestY, np.zeros(sampleTrainX.size))
        print "Without biasing Technique: c = ", maxC, "gamma = ", maxG, "modelSize = ", modelSize, "maxacc = ", maxacc
        noBiasacc.append(maxacc)
         
        # # with ulsif 
        result = densratio(sampleTrainX, sampleTestX)  
        weights = result.compute_density_ratio(sampleTrainX)
        maxacc, maxC, maxG = getAccuracy(sampleTrainX, sampleTrainY, sampleTestX, sampleTestY, weights)
        print "With Ulsif: c = ", maxC, "gamma = ", maxG, "modelSize = ", modelSize, "maxacc = ", maxacc            
        ulsifacc.append(maxacc)
         
        # # with pykliep
        kliep = DensityRatioEstimator()
        kliep.fit(sampleTrainX, sampleTestX) 
        weights = kliep.predict(sampleTrainX)
        maxacc, maxC, maxG = getAccuracy(sampleTrainX, sampleTrainY, sampleTestX, sampleTestY, weights)
        print "With pyklieb: c = ", maxC, "gamma = ", maxG, "modelSize = ", modelSize, "maxacc = ", maxacc       
        kliepacc.append(maxacc)
          
        # # with kmm
        gammab = [0.005, 0.7, 7]
        beta = kmm.kmm.getBeta(sampleTrainX, sampleTestX, gammab)
        weights = np.array(beta)
        maxacc, maxC, maxG = getAccuracy(sampleTrainX, sampleTrainY, sampleTestX, sampleTestY, weights)
        print "With Kmm: c = ", maxC, "gamma = ", maxG, "modelSize = ", modelSize, "maxacc = ", maxacc            
        kmmacc.append(maxacc)
         
    print noBiasacc
    print kmmacc
    print kliepacc
    print ulsifacc
    
    plot(noBiasacc, kmmacc, kliepacc, ulsifacc)  
       
main()        
        
