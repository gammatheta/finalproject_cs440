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
PRINT = True
STREAK_REQUIREMENT = 0.75
TRAIN_VALUE = 0.9

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
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels)
    self.weights = weights
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels, filename):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    
    self.features = list(trainingData) # could be useful later
    # print(f"size of list in feature: {len(self.features[0])}")
    
    allkeys = self.features[0].keys()
    
    using_prime = True if filename == 'prime' else False
    if using_prime:
      print("using prime")
      filename = 'primefaceweights.txt' if len(self.legalLabels) == 2 else 'primedigitweights.txt'
      self.weights = util.Counter.parse_weights(filename)
    else:
      print("not using prime")
      for label in self.legalLabels:
        for key in allkeys:
          weight = random.random()
          self.weights[label][key] = weight
        self.weights[label]['bias'] = 1

    # streak = 0    
    # print(f'Running for {self.max_iterations} minutes')
    t_end = time.time() + (60 * self.max_iterations)
    start_time = time.time()
    isTimeUp = False
    training_rate = 1
    iterations = 0
    corrections = 0
    # print(f"feature size: {len(self.features)}")
    # print(self.features[0])
    while time.time() < t_end:
      # print(f'beginning iteration {iteration}...')
      guesses = self.classify(self.features)
      
      '''
      if time.time() >= t_end:
          isTimeUp = True
          break
      '''
      for i in range(len(self.features)):
        # print(f"Beginning feature {i}")
        # "*** YOUR CODE HERE ***"
        # print(f"feature #{i};")
        
        trainingData_list = self.features[i]
        ans = trainingLabels[i]
        guess = guesses[i]
        # print(f"ml guessed: {guess}, ans was: {ans}")
        if ans != guess:
          # print("incorrect")
          self.weights[guess] -= trainingData_list.multiplyAll(training_rate)
          self.weights[guess]['bias'] = self.weights[guess]['bias'] - 1
          self.weights[ans] += trainingData_list.multiplyAll(training_rate)
          self.weights[ans]['bias'] = self.weights[ans]['bias'] + 1
          corrections += 1
        # print(f"finished feature {i}")
      # print()
      # print(f"Corrections made during iteration {iterations}: {corrections}")
      corrections = 0
      training_rate *= 0.95
      iterations += 1
      if isTimeUp:
        break
    correct = [guesses[i] == trainingLabels[i] for i in range(len(trainingLabels))].count(True)
    end_time = time.time()
    time_diff = int(end_time - start_time)
    print(f"Training finished after {iterations} iterations with {correct} correct out of {len(self.features)}")
    print(f"Time elapsed: {time_diff} seconds")

    '''
    with open(filename, 'w') as f:
      for label, weights in self.weights.items():
        f.write(f"{label}:\n")
        for key,weight in weights.items():
          f.write(f"{key};{weight};")
        f.write("\n")
    f.close
    '''
    

    # for iteration in range(self.max_iterations):
    #   print("Starting iteration ", iteration, "...")
    #   for i in range(len(self.features)):
    #       "*** YOUR CODE HERE ***"
    #       # util.raiseNotDefined()
    #       # print(i)
    #       trainingData_list = self.features[i]
    #       ans = trainingLabels[i]
    #       guess = self.classify(self.features)[i]
    #       if ans != guess:
    #         self.weights[guess] = self.weights[guess] - trainingData_list
    #         self.weights[ans] = self.weights[ans] + trainingData_list

    # print("weights: " + str(self.weights))
    # print("trained!")
    
  def classify(self, data):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    #print(type(data))
    guesses = []
    for datum in data:
      #print(type(datum))
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = (self.weights[l] * datum) + self.weights[l]['bias']
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

  