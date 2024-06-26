# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness 
# code for data classification

# import mostFrequent
# import naiveBayes
import collections
import numpy as np
import perceptron
import neuralnetwork
import samples
import sys
import util

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()
  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  # print(f"printing dataum:\n{datum}")
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  correct = np.array([guesses[i] == testLabels[i] for i in range(len(testLabels))])
  testing = True
  while testing:
    usr_input = int(input("Test specific image? (pick a number between 0 to 99) or -1 to end this instance of testing: "))
    if usr_input == -1:
      print("Testing instance was ended")
      print("--------------------------------------------------------")
      break

    print(rawTestData[usr_input])
    if guesses[usr_input] == testLabels[usr_input]:
      print(f"Program prediction {guesses[usr_input]} was correct, number was {testLabels[usr_input]}.")
    else:
      print(f"Program prediction {guesses[usr_input]} was incorrect, number was {testLabels[usr_input]}")

    print("========================================================")
           

## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which 
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print("new features:", pix)
            continue
      print(image)

def default(str):
  return str + ' [Default: %default]'

def readCommand(argv):
  "Processes the command used to run from the command line."
  from optparse import OptionParser  
  parser = OptionParser(USAGE_STRING)
  
  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['perceptron', 'neural'], default='perceptron')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
  parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=5, type="int")
  parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")
  parser.add_option('-p', '--prime', help=default("Whether to use prime weight files"), choices=['', 'prime'], default='')

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {}
  
  # Set up variables according to the command line input.
  print("Doing classification")
  print("--------------------")
  print("data:\t\t" + options.data)
  print("classifier:\t\t" + options.classifier)
  
  print("training set size:\t" + str(options.training))
  if(options.data=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
    featureFunction = basicFeatureExtractorDigit
  elif(options.data=="faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
    featureFunction = basicFeatureExtractorFace
  else:
    print("Unknown dataset", options.data)
    print(USAGE_STRING)
    sys.exit(2)
    
  if(options.data=="digits"):
    legalLabels = range(10)
  else:
    legalLabels = range(2)
    
  if options.training <= 0:
    print(f"Training set size should be a positive integer (you provided: {options.training})")
    print(USAGE_STRING)

  if(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
  # print("iterations: " + str(options.iterations))
  elif(options.classifier == "neural"):
    classifier = neuralnetwork.NeuralNetworkClassifier(legalLabels, options.iterations)

    print("iterations: " + str(options.iterations) + "mins")
  elif(options.classifier == "neural"):
    classifier = neuralnetwork.NeuralNetworkClassifier(legalLabels,options.iterations)
    print("iterations: " + str(options.iterations) + "mins")
  else:
    print("Unknown classifier:", options.classifier)
    print(USAGE_STRING)
    
    sys.exit(2)

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage
  print(f"options: {options}")
  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """

# Main harness code

def runClassifier(args, options):

  featureFunction = args['featureFunction']
  classifier = args['classifier']
  printImage = args['printImage']
      
  # Load data  
  numTraining = options.training
  numTest = options.test

  if(options.data=="faces"):
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
    rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
    rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
  else:
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)
  
  filename = 'prime' if options.prime == 'prime' else ''
  
  # Extract features
  print("Extracting features...")
  trainingData = map(featureFunction, rawTrainingData)
  validationData = map(featureFunction, rawValidationData)
  testData = map(featureFunction, rawTestData)
  
  # Conduct training and testing
  print("Training...")
  classifier.train(trainingData, trainingLabels, validationData, validationLabels, filename)
  print("Validating...")
  guesses = classifier.classify(validationData)
  if isinstance(guesses, np.ndarray): correct = [np.all(guesses[i] == validationLabels[i]) for i in range(len(validationLabels))].count(True)
  else: correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
  print(f"{str(correct)} correct out of {str(len(validationLabels))} ({100.0 * correct / len(validationLabels):.1f}).")
  print("Testing...")
  guesses = classifier.classify(testData)
  if isinstance(guesses, np.ndarray): correct = [np.all(guesses[i] == testLabels[i]) for i in range(len(testLabels))].count(True)
  else: correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print(f"{str(correct)} correct out of {str(len(testLabels))} ({100.0 * correct / len(testLabels):.1f}).")
  analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)
  
  '''
  # do odds ratio computation if specified at command line
  if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
    label1, label2 = options.label1, options.label2
    features_odds = classifier.findHighOddsFeatures(label1,label2)
    if(options.classifier == "naiveBayes" or options.classifier == "nb"):
      string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
    else:
      string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)    
      
    print(string3)
    printImage(features_odds)
'''

  if((options.weights) & (options.classifier == "perceptron")):
    for l in classifier.legalLabels:
      features_weights = classifier.findHighWeightFeatures(l)
      print ("=== Features with high weight for label %d ==="%l)
      printImage(features_weights)
  elif((options.weights) & (options.classifier == "neural")):
    for l in classifier.legalLabels:
      features_weights = classifier.findHighWeightFeatures(l)
      print ("=== Features with high weight for label %d ==="%l)
      printImage(features_weights)

  if((options.weights) & (options.classifier == "neural")):
    for l in classifier.legalLabels:
      features_weights = classifier.findHighWeightFeatures(l)
      print ("=== Features with high weight for label %d ==="%l)
      printImage(features_weights)

if __name__ == '__main__':
  # Read input
  if len(sys.argv) <= 1:
    testing = True
    while testing:
      classifier = input("Which classifier would you like to use? (perceptron or neural)\n").lower()
      imagetype = input("faces or digits?\n").lower()
      dataset = input("How many images would you like to test for the dataset? (1 to 5000 for digits / 1 to 450 for faces)\n").lower()
      iterations = input("How many epochs would you like to run? (number of loops that will be done in training)\n").lower()
      prime = input("Would you like to use the prime weights? Yes or No (Best case test weights that were generated from a prior run and stored for use in future runs of this program)\n").lower()
      filename = 'prime' if prime == 'yes' else ''
      
      sys.argv = ['dataClassifier.py', '-c', classifier, '-d', imagetype, '-i', iterations, '-t', dataset, '-p', filename]
      print(f"argv: {sys.argv}")
      args, options = readCommand(sys.argv[1:])
      print(f'options: {options}')
      runClassifier(args, options)

      check = input("Would you like to rerun the program to test another instance? (Yes or No)\n").lower()
      testing = True if check == 'yes' else False
  else:
    args, options = readCommand(sys.argv[1:])
    runClassifier(args, options)

  print("Program terminated.\n")  