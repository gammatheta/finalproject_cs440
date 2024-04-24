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
READ = False
LAMBDA = 0.03
ALPHA = 0.01

class NeuralNetworkClassifier:
    def __init__(self, legalLabels, max_iterations):
        self.labels = legalLabels
        self.max_iterations = max_iterations
        
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
        self.layerOneWeights = np.random.rand(self.hiddenUnits,self.inputUnits+1)
        self.layerTwoWeights = np.random.rand(self.outputUnits,self.hiddenUnits+1)

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def backpropagate():
        pass
    
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        trainingData will come as a map of util.Counter() -> convert to list of util.Counter()
        listOfAllDataPoints is list of lists where each sublist is a list of all the datapoint values of a single image
            i.e: [{(0,0) = 1 , (0,1) = 0}, ... ] -> [[1,0], ...]
        
        """
        # read from weights file to do training
        if READ:
            pass

        self.features = list(trainingData)
        listOfAllDataPoints = []
        for i in range(len(self.features)):
            listOfAllDataPoints.append(list(self.features[i].values()))
        npArrOfInputDataPoints = []
        for points in listOfAllDataPoints:
            npArrOfInputDataPoints.append(np.array([1]+points).reshape(-1,1))

        print(f'Running for {self.max_iterations} minutes')
        t_end = time.time() + (60 * self.max_iterations)
        isTimeUp = False
        iteration = 0
        
        while True:
            self.D1 = np.zeros((self.hiddenUnits,self.hiddenUnits+1))
            self.D2 = np.zeros((self.outputUnits,self.hiddenUnits+1))
            self.gradientLayerOne = np.zeros((self.hiddenUnits,self.hiddenUnits+1))
            self.gradientLayerTwo = np.zeros((self.outputUnits,self.hiddenUnits+1))
            for i in range(len(npArrOfInputDataPoints)):
                if time.time() >= t_end:
                    isTimeUp = True
                    break
                ans = trainingLabels[i]
                inputLayer = npArrOfInputDataPoints[i]
                # calculate hidden layer
                hiddenLayer = np.dot(self.layerOneWeights, inputLayer)
                hiddenLayer = 1/(1+np.exp(-1*hiddenLayer))
                hiddenLayer = np.insert(hiddenLayer, 0, 1).reshape(-1,1)
                # calculate output layer
                outputLayer = np.dot(self.layerTwoWeights, hiddenLayer)
                outputLayer = 1/(1 + np.exp(-1*outputLayer))
                outputLayer = outputLayer.reshape(-1,1)
                # take a guess
                guess = np.argmax(outputLayer)
                
                if ans != guess:
                    answerVector = np.zeros(self.outputUnits).reshape(-1,1)
                    answerVector[ans] = 1
                    deltaLayerThree = outputLayer - answerVector
                    # print(f'shape of deltaLayerThree is {deltaLayerThree.shape}')
                    # print(f'shape of layer2weights is {self.layerTwoWeights.shape}')
                    # print(f'shape of layer1weights is {self.layerOneWeights.shape}')
                    # print(f'shape of hiddenlayer is {hiddenLayer.shape}')
                    deltaLayerTwo = np.dot(self.layerTwoWeights.T,deltaLayerThree)
                    # print(f'shape of deltaLayer2 is {deltaLayerTwo.shape}')
                    gprime = np.multiply(hiddenLayer, (1-hiddenLayer))
                    # print(f'shape of gprime is {gprime.shape}')
                    deltaLayerTwo = np.multiply(deltaLayerTwo,gprime)
                    deltaLayerTwo = deltaLayerTwo[1:]
                    # print(f'shape of deltaLayer2 is {deltaLayerTwo.shape}')

                    self.gradientLayerOne = self.gradientLayerOne + np.dot(deltaLayerTwo,inputLayer.T)
                    self.gradientLayerTwo = self.gradientLayerTwo + np.dot(deltaLayerThree,hiddenLayer.T)
                    # print(f'shape of gradient1 is {self.gradientLayerOne.shape}')
                    # print(f'shape of gradient2 is {self.gradientLayerTwo.shape}')
                    # print(f'Incorrect answer: ans was {ans} and guess was {guess} from guessing {outputLayer}')

            # compute avg regularized gradient
            for i in range(self.D1.shape[0]):
                for j in range(self.D1.shape[1]):
                    if j == 0:
                        self.D1[i, j] =  1/(len(npArrOfInputDataPoints)) * self.gradientLayerOne[i,j] 
                    else:
                        self.D1[i, j] =  (1/(len(npArrOfInputDataPoints)) * self.gradientLayerOne[i,j]) + (LAMBDA * self.layerOneWeights[i,j])
            
            for i in range(self.D2.shape[0]):
                for j in range(self.D2.shape[1]):
                    if j == 0:
                        self.D2[i, j] =  1/(len(npArrOfInputDataPoints)) * self.gradientLayerTwo[i,j] 
                    else:
                        self.D2[i, j] =  (1/(len(npArrOfInputDataPoints)) * self.gradientLayerTwo[i,j]) + (LAMBDA * self.layerTwoWeights[i,j])
            self.layerOneWeights = self.layerOneWeights - (ALPHA * self.D1)
            self.layerTwoWeights = self.layerTwoWeights - (ALPHA * self.D2)
            if isTimeUp:
                break
        
        # np.savetxt('digitweights1-nn.txt',self.layerOneWeights,fmt='%.4f',delimiter=',')
        # np.savetxt('digitweights2-nn.txt',self.layerTwoWeights,fmt='%.4f',delimiter=',')
    

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.
        
        Recall that a datum is a util.counter... 
        """
        guesses = []
        
        for datum in data:
            inputLayer = np.array([1]+list(datum.values())).reshape(-1,1)
            # calculate hidden layer
            hiddenLayer = np.dot(self.layerOneWeights, inputLayer)
            hiddenLayer = 1/(1+np.exp(-1*hiddenLayer))
            hiddenLayer = np.insert(hiddenLayer, 0, 1).reshape(-1,1)
            # calculate output layer
            outputLayer = np.dot(self.layerTwoWeights, hiddenLayer)
            outputLayer = 1/(1 + np.exp(-1*outputLayer))
            outputLayer = outputLayer.reshape(-1,1)
            # take a guess
            guess = np.argmax(outputLayer)
            guesses.append(guess) 
        return guesses