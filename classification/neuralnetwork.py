import util
import math
import numpy as np
import time
import random
import string

def activation_function():
    pass

class NeuralNetworkClassifier:
    def __init__(self, legalLabels, max_iterations):
        self.labels = legalLabels
        self.max_iterations = max_iterations
        self.weights = {}
    

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        pass

    def classify(self, data):
        pass