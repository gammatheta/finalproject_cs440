import random
import time
import numpy as np
import collections
PRINT = True
STREAK_REQUIREMENT = 0.75
TRAIN_VALUE = 0.9

class NeuralNetworkClassifier:
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "neuralnetwork"
        self.max_iterations = max_iterations
        self.weights1 = np.array([])
        self.weights2 = np.array([])

    def train(self, trainingData, trainingLabels, validationData, validationLabels):        
        training_arr = self.counter_to_array(list(trainingData))
        # print(f"counter to array returned an array in shape of: {training_arr.shape}")
        self.weights1 = np.random.rand(training_arr.shape[0],training_arr.shape[1])
        self.weights2 = np.random.rand(10,training_arr.shape[0])
        
        # print(f"activation returned hidden layer shape: {hidden_layer.shape}")
        # self.classify(hidden_layer)
        # print(training_arr)
        countdown = time.time() + (self.max_iterations * 60)
        timer = True
        while timer:
            guesses = self.classify(training_arr)
            if time.time() >= countdown:
                timer = False
        # np.savetxt('neuronguesses.txt', guesses, fmt='%i', delimiter=',')

    def classify(self, datum):
        if not isinstance(datum, np.ndarray):
            # print("incompatible type with datum")
            datum = self.counter_to_array(datum)
        # print(f"shape of data passed in: {datum.shape}")
        # if type(datum)
        v =  np.dot(self.weights1, datum.T)
        hidden_layer = np.maximum(0,v)
        # np.savetxt('neuronhidden.txt', hidden_layer, fmt='%f', delimiter=' ')

        v = np.dot(self.weights2, hidden_layer) # potentially incorrect due to self.weights2's shape and value placement
        output_layer = np.maximum(0,v).T
        
        guesses = np.argmax(output_layer, axis=1)

        return guesses
    '''
    def classify(self, datum):
        print("made it to classify")
        exit()
    '''
    def counter_to_array(self, datum):
        datum_list = []
        for data in datum:
            datum_list.append(np.array(list(data.values())))
        
        return np.array(datum_list)