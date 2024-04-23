import util
import math
import numpy as np
import time
import random
import string

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70
STREAK_REQUIREMENT = 0.75
TRAIN_VALUE = 0.9

class NeuralNetworkClassifier:
    def __init__(self, legalLabels, max_iterations):
        self.labels = legalLabels
        self.max_iterations = max_iterations
        self.layerOneWeights = {}
        self.layerTwoWeights = {}
        
        # determine amount of units each layer has
        if len(legalLabels) == 2:
            self.inputUnits = FACE_DATUM_HEIGHT * FACE_DATUM_WIDTH
            self.hiddenUnits = self.inputUnits
            self.outputUnits = 2
        else:
            self.inputUnits = DIGIT_DATUM_HEIGHT * DIGIT_DATUM_WIDTH
            self.hiddenUnits =  self.inputUnits
            self.outputUnits = 10

        # set random weights for each layer 
        for label in legalLabels:
            self.layerOneWeights[label] = np.random.rand(self.hiddenUnits,self.inputUnits+1)
            self.layerTwoWeights[label] = np.random.rand(self.outputUnits,self.hiddenUnits+1)

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights
    
    def activation_function():
        pass
    
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        trainingData will come as a map of util.Counter() -> convert to list of util.Counter()
        listOfAllDataPoints is list of lists where each sublist is a list of all the datapoint values of a single image
            i.e: [{(0,0) = 1 , (0,1) = 0}, ... ] -> [[1,0], ...]
        
        """
        # filename = 'faceweights.txt' if len(self.legalLabels) == 2 else 'digitweights.txt'
        # with open('initial_weights.txt', 'w') as f:
        #     for label, weight in self.weights.items():
        #         f.write(f"{label}:\n{weight}\n")
        #     f.close

        self.features = list(trainingData)
        listOfAllDataPoints = []
        for i in range(len(self.features)):
            listOfAllDataPoints.append(self.features[i].values())
        npArrOfInputDataPoints = []
        for points in listOfAllDataPoints:
            npArrOfInputDataPoints.append(np.array([1]+points).reshape(-1,1))

        print(f'Running for {self.max_iterations} minutes')
        t_end = time.time() + (60 * self.max_iterations)
        isTimeUp = False
        iteration = 0

        while True:
            for i in range(len(npArrOfInputDataPoints)):
                if time.time() >= t_end:
                    isTimeUp = True
                    break
                ans = trainingLabels[i]
                inputLayer = npArrOfInputDataPoints[i]
                hiddenLayer = np.dot()
                guess = guesses[i]
                
                if ans != guess:
                    pass
            if isTimeUp:
                break

        # print("trained!")
        # with open(filename, 'w') as f:
        #     for label, weights in self.weights.items():
        #         f.write(f"{label}:\n")
        #         for key,weight in weights.items():
        #             f.write(f"{key};{weight};")
        #             f.write("\n")
        #     f.close
        # with open("initial_weights.txt") as i, open(filename) as w: 
        #     initial = i.readlines()
        #     after = w.readlines()
        #     if initial == after:
        #         print("no update in weights after iterations")
        #     else:
        #         print("weights were updated")
        #     i.close()
        #     w.close()

    def classify(self, data):
        pass