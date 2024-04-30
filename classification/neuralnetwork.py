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
        if max_iterations == 0:
            self.max_iterations = 1
        else:
            self.max_iterations = max_iterations
        self.bias1 = 0.1
        self.bias2 = 0.2
        self.hidden_layer = []
        self.output_layer = []
        self.weights1 = np.array([])
        self.weights2 = np.array([])

    def train(self, trainingData, trainingLabels, validationData, validationLabels, filename):        
        training_arr = self.counter_to_array(list(trainingData)) # converts training data into a numpy array to perform numpy operations with data
        datasize = training_arr.shape[0]
        pixel_size = training_arr.shape[1]
        hidden_size = 512 # amount of neurons for the hidden layer
        output_size = len(self.legalLabels)
        bin_tlabels = self.to_binary(trainingLabels)

        using_prime = True if len(filename) > 0 else False

        # initialization of weights
       
        if using_prime:
            print("using prime")
            weights1file = 'primeneuralfacesweights1.txt' if training_arr.shape[1] > 784 else 'primeneuraldigitsweights1.txt'
            weights2file = 'primeneuralfacesweights2.txt' if training_arr.shape[1] > 784 else 'primeneuraldigitsweights2.txt'
            self.weights1 = np.loadtxt(weights1file, delimiter=' ')
            self.weights2 = np.loadtxt(weights2file, delimiter=' ')
        else:
            print("not using prime")
            self.weights1 = np.random.randn(hidden_size,pixel_size) * np.sqrt(2 / hidden_size)
            self.weights2 = np.random.randn(output_size,hidden_size) * np.sqrt(2 / hidden_size)

        # countdown = time.time() + (self.max_iterations * 60)
        end_time = 'inf'
        initial_time = time.time()
        gradients1 = np.zeros((hidden_size, pixel_size))
        gradients2 = np.zeros((output_size, hidden_size))
        timer = True
        streak = 0
        correct = 0
        iterations = 0
        while iterations < self.max_iterations:
            guesses = self.classify(training_arr)

            delta3 = self.output_layer - bin_tlabels
            delta2 = np.dot(self.weights2.T, delta3) * (self.hidden_layer * (1 - self.hidden_layer))

            gradients2 = gradients2 + np.dot(delta3, self.hidden_layer.T)
            gradients1 = gradients1 + np.dot(delta2, training_arr)
            
            avg_grad1 = 1/datasize * gradients1 + regularization * self.weights1
            avg_grad2 = 1/datasize * gradients2 + regularization * self.weights2
            self.weights1 = self.weights1 - learning_rate * avg_grad1
            self.weights2 = self.weights2 - learning_rate * avg_grad2
            
            # correct = [np.all(guesses[i] == trainingLabels[i]) for i in range(len(trainingLabels))].count(True)
            iterations += 1
            '''
            if i % 1000 == 0: print(f"Correct guesses: {correct} out of {len(training_arr)} after {i} iterations")
            if (correct >= len(training_arr) * 0.99): streak += 1
            if time.time() >= countdown or streak >= 100: 
                
                timer = False
            '''
        end_time = time.time()
        time_diff = int(end_time - initial_time)
        # correct = [np.all(guesses[i] == trainingLabels[i]) for i in range(len(trainingLabels))].count(True)

        # print(f"Training finished after {iterations} iterations with {correct} correct out of {len(training_arr)} correct predictions.")
        print(f"Time elapsed: {time_diff} seconds")
        # np.savetxt(weights1file, self.weights1, fmt='%f', delimiter=' ')
        # np.savetxt(weights2file, self.weights2, fmt='%f', delimiter=' ')

    def classify(self, datum):
        if not isinstance(datum, np.ndarray):
            datum = self.counter_to_array(datum)
        layer = 2
        self.hidden_layer = self.activation_func(datum, self.weights1, self.bias1, layer)
        layer += 1
        self.output_layer = self.activation_func(self.hidden_layer, self.weights2, self.bias2, layer)

        guesses = np.argmax(self.output_layer, axis=0)
        return guesses
    
    def activation_func(self, data, weights, bias, layer):
        """
        Performs Sigmoid function and returns resulting matrix.
        """

        dot = np.dot(weights, data.T) + bias if layer < 3 else np.dot(weights, data) + bias
        v = np.exp(-dot)
        return 1 / (1 + v)

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