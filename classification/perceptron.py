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
STREAK_REQUIREMENT = 0.7
TRAIN_VALUE = 1
TRAIN_VALUE_MODIFIER = 0.95

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
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
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
    
    allkeys = self.features[0].keys()
    # for key in allkeys:
    #   for label in self.legalLabels:
    #     weight = random.random()
    #     self.weights[label][key] = weight
    # for label in self.legalLabels:
    #   self.weights[label]['bias'] = 1

    for label in self.legalLabels:
      for key in allkeys:
        weight = random.random()
        self.weights[label][key] = weight
      self.weights[label]['bias'] = 1 
      
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
    # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

    streak = 0    
    print(f'Running for {self.max_iterations} minutes')
    t_end = time.time() + (60 * self.max_iterations)
    isTimeUp = False
    amtNeeded = int(STREAK_REQUIREMENT * len(self.features))
    learnRate = TRAIN_VALUE
    # iteration = 0
    # while True:
    #   streak = 0
      
    #   for i in range(len(self.features)):
    #     "*** YOUR CODE HERE ***"
    #     if time.time() >= t_end or streak == amtNeeded:
    #       isTimeUp = True
    #       break
    #     trainingData_list = self.features[i]
    #     ans = trainingLabels[i]
    #     guess = self.classify(self.features)[i]
    #     if ans != guess:
    #       self.weights[guess] = self.weights[guess] - trainingData_list.multiplyAll(learnRate)
    #       self.weights[guess]['bias'] = self.weights[guess]['bias'] - 1
    #       self.weights[ans] = self.weights[ans] + trainingData_list.multiplyAll(learnRate)
    #       self.weights[ans]['bias'] = self.weights[ans]['bias'] + 1
    #     else:
    #       streak += 1
    #   learnRate = TRAIN_VALUE_MODIFIER * learnRate
      
    #   if isTimeUp:
    #     break


    for iteration in range(self.max_iterations):
      print("Starting iteration ", iteration, "...")
      for i in range(len(self.features)):
          "*** YOUR CODE HERE ***"
          # util.raiseNotDefined()
          # print(i)
          trainingData_list = self.features[i]
          ans = trainingLabels[i]
          guess = self.classify(self.features)[i]
          if ans != guess:
            self.weights[guess] = self.weights[guess] - trainingData_list.multiplyAll(learnRate)
            self.weights[guess]['bias'] = self.weights[guess]['bias'] - 1
            self.weights[ans] = self.weights[ans] + trainingData_list.multiplyAll(learnRate)
            self.weights[ans]['bias'] = self.weights[ans]['bias'] + 1
      learnRate = TRAIN_VALUE_MODIFIER * learnRate
    # print("weights: " + str(self.weights))
    # print("trained!")
    
    with open('weights.txt', 'w') as f:
      for label, weight in self.weights.items():
        f.write(f"{label}:\n{weight}\n")
      f.close

  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
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

