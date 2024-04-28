import random
import time
import numpy as np
import collections
PRINT = True
learning_rate = 0.01
regularization = 0.01

class NeuralNetworkClassifier:
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "neuralnetwork"
        self.max_iterations = max_iterations
        self.bias1 = 0.1
        self.bias2 = 0.2
        self.hidden_layer = []
        self.output_layer = []
        self.weights1 = np.array([])
        self.weights2 = np.array([])

    def train(self, trainingData, trainingLabels, validationData, validationLabels):        
        training_arr = self.counter_to_array(list(trainingData))
        datasize = training_arr.shape[0]
        pixel_size = training_arr.shape[1]
        hidden_size = 512
        output_size = len(self.legalLabels)
        self.weights1 = np.random.randn(hidden_size,pixel_size) * np.sqrt(2 / hidden_size)
        self.weights2 = np.random.randn(output_size,hidden_size) * np.sqrt(2 / hidden_size)
        bin_tlabels = self.to_binary(trainingLabels)        

        # print(f"shape of weights1: {self.weights1.shape}")
        # print(f"shape of weights2: {self.weights2.shape}")
        # print(f"activation returned hidden layer shape: {hidden_layer.shape}")
        # self.classify(hidden_layer)
        # print(training_arr)
        countdown = time.time() + (self.max_iterations * 60)
        gradients1 = np.zeros((hidden_size, pixel_size))
        gradients2 = np.zeros((output_size, hidden_size))
        timer = True
        correct = 0
        i = 0
        while timer:
            guesses = self.classify(training_arr)
            '''
            np.savetxt('neuronhidden.txt', self.hidden_layer, fmt='%f', delimiter=' ')
            np.savetxt('neuronoutput.txt', self.output_layer, fmt='%f', delimiter=' ')
            '''
            # print(f"shape of weights1: {self.weights1.shape}")
            # print(f"shape of weights2: {self.weights2.shape}")
            # print(f"shape of hidden layer: {self.hidden_layer.shape}")
            # print(f"shape of output layer: {self.output_layer.shape}")

            # output_loss = self.cost_func(self.output_layer, bin_tlabels)
            # np.savetxt('neuronloss.txt', loss, fmt='%f', delimiter=' ')
            delta3 = self.output_layer - bin_tlabels
            delta2 = np.dot(self.weights2.T, delta3) * (self.hidden_layer * (1 - self.hidden_layer))

            # print(f"shape of delta3: {delta3.shape}")
            # print(f"shape of delta2: {delta2.shape}")
            

            gradients2 = gradients2 + np.dot(delta3, self.hidden_layer.T)
            gradients1 = gradients1 + np.dot(delta2, training_arr)
            
            # errors = self.compute_errors()
            # print(delta)
            avg_grad1 = 1/datasize * gradients1 + regularization * self.weights1
            avg_grad2 = 1/datasize * gradients2 + regularization * self.weights2
            self.weights1 = self.weights1 - learning_rate * avg_grad1
            self.weights2 = self.weights2 - learning_rate * avg_grad2
            
            correct = [np.all(guesses[i] == trainingLabels[i]) for i in range(len(trainingLabels))].count(True)
            i += 1
            if i % 100 == 0:
                print(f"Correct guesses: {correct} out of {len(training_arr)} after {i} iterations")
            if time.time() >= countdown or correct == len(training_arr):
                timer = False
        np.savetxt("neurondelta.txt", delta3, fmt='%.4f', delimiter=' ')
        np.savetxt("neurongradients1.txt", gradients1, fmt='%.4f', delimiter=' ')
        np.savetxt("neurongradients2.txt", gradients2, fmt='%.4f', delimiter=' ')
        np.savetxt("neuronweights1.txt", self.weights2, fmt='%.4f', delimiter=' ')
        np.savetxt("neuronweights2.txt", self.weights2, fmt='%.4f', delimiter=' ')
        # np.savetxt('neuronguesses.txt', guesses, fmt='%i', delimiter=',')

    def classify(self, datum):
        if not isinstance(datum, np.ndarray):
            # print("incompatible type with datum")
            datum = self.counter_to_array(datum)
        # print(f"shape of data passed in: {datum.shape}")
        # if type(datum)
        self.hidden_layer = self.activation_func(datum, self.weights1, self.bias1)
 
        self.output_layer = self.activation_func(self.hidden_layer, self.weights2, self.bias2) # potentially incorrect due to self.weights2's shape and value placement
        # print(f"shape of neuron output: {self.output_layer.shape}")

        guesses = np.argmax(self.output_layer, axis=0)
        return guesses
    
    def activation_func(self, data, weights, bias):
        """
        Performs Sigmoid function and returns resulting matrix.
        """

        v = np.dot(weights, data.T) + bias if np.all(data != self.hidden_layer) else np.dot(weights, data) + bias
        '''
        if np.all(weights == self.weights1) and data.shape[0] > 100:
            np.savetxt('v1.txt', v, fmt='%f', delimiter=' ')
        elif np.all(weights == self.weights2) and data.shape[0] > 100:
            np.savetxt('v2.txt', v, fmt='%f', delimiter=' ')
        '''
        return 1 / (1 + np.exp(-v))
    
    def cost_func(self, data, labels):
        return labels * np.log(data) + (1 - labels) * np.log(1 - data)

    def compute_gradients(self):
        print("compute_gradients incomplete function, exiting")
        exit()

    def to_binary(self, data):
        """
        Converts label data passed in to a binary matrix
        """
        return np.eye(len(self.legalLabels))[data].T
    
    def counter_to_array(self, datum):
        """
        Performs Counter(dict) to numpy array conversions and returns the array.
        """
        datum_list = []
        for data in datum:
            datum_list.append(np.array(list(data.values())))
        
        return np.array(datum_list)