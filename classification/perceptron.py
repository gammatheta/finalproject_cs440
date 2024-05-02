# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util
import random
import time
import numpy as np
PRINT = True
STREAK_REQUIREMENT = 0.75
TRAIN_VALUE = 0.9
TRAINING_RATE = float(1)

class PerceptronClassifier:
  """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.weights = np.array([self.legalLabels])
    self.bias = np.array([])

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels)
    self.weights = weights
      
  def train(self, trainingData, trainingLabels, validationData, validationLabels, filename):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    
    

    self.features = self.counter_to_array(list(trainingData)) # could be useful later
    height = self.features.shape[0]
    width = self.features.shape[1]

    print(f"shape of features array: {self.features.shape}\n{self.features}")
    
    using_prime = True if filename == 'prime' else False
    if using_prime:
      print("using prime")
      filename = 'primefaceweights.txt' if len(self.legalLabels) == 2 else 'primedigitweights.txt'
      self.weights = np.loadtxt(filename, delimiter=' ')
    else:
      print("not using prime")
      self.weights = np.random.rand(width, height)
      self.bias = np.ones((height, 1))

    print(f"bias: {self.bias}")
    # streak = 0    
    # print(f'Running for {self.max_iterations} minutes')
    # t_end = time.time() + (60 * self.max_iterations)
    start_time = time.time()
    isTimeUp = False
    iterations = 0
    corrections = 0
    # print(f"feature size: {len(self.features)}")
    # print(self.features[0])
    while iterations < self.max_iterations:
      # print(f'beginning iteration {iteration}...')
      guesses = self.classify(self.features)
      
      '''
      if time.time() >= t_end:
          isTimeUp = True
          break
      '''
      for i in range(len(self.features)):
        trainingData_list = self.features[i]
        ans = trainingLabels[i]
        guess = guesses[i]
        # print(f"ml guessed: {guess}, ans was: {ans}")
        if ans != guess:
          # print("incorrect")
          self.weights[guess] -= trainingData_list * TRAINING_RATE
          self.bias[guess] = self.weights[guess]['bias'] - 1
          self.weights[ans] += trainingData_list * TRAINING_RATE
          self.bias[ans] = self.weights[ans]['bias'] + 1
          corrections += 1

      corrections = 0
      TRAINING_RATE *= 0.95
      iterations += 1
      if isTimeUp:
        break
    correct = [guesses[i] == trainingLabels[i] for i in range(len(trainingLabels))].count(True)
    end_time = time.time()
    time_diff = int(end_time - start_time)
    print(f"Training finished after {iterations} iterations with {correct} correct out of {len(self.features)}")
    print(f"Time elapsed: {time_diff} seconds")
    
  def classify(self, data):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is an array... 
    """
    guesses = []
    for datum in data:
      for i in len(self.legalLabels):
        vectors = (self.weights[i] * datum) + self.weights[l]['bias']
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    featuresWeights = []

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresWeights

  def counter_to_array(self, datum):
    """
    Performs Counter(dict) to numpy array conversions and returns the array.
    """
    datum_list = []
    for data in datum:
        datum_list.append(np.array(list(data.values())))
    
    return np.array(datum_list)