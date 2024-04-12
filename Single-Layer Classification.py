import numpy as np
# find a lib to extract the photo into an array

# extract image and return features from the extracted photo
def extract_image(filename):
    # extract the photo and convert it into an np array, 1 = black pixel, 0 = white pixel
    # divide the array up into a set of features with each value being number of black pixels within feature
    return None

def perceptron_faces(filename):
    # initialize values for features for image(X_i)
    # initialize weights randomly (random 1 to 10?)
    # initialize bias value w_0

    # run scoring calculation using (summation of weights * features) + bias
    # if score >= 0 and y = 1 or score < 0 and y = 0: do nothing and move onto the next image
    # if score >= 0 and y = 0: punish weights that contributed to error
    #   w_n = w_n - phi_n and w_n = w_n - phi_n for rest of weights
    #   w_0 = w_0 - 1
    # if score < 0 and y = face: 
    #   w_n = w_n + phi_n and w_n = w_n + phi_n for rest of weights
    #   w_0 = w_0 + 1
    # then move onto next image(X_i)
    return None

def perceptron_multiclass(filename):
    # initialize values for features for image.X_i
    # initialize weights randomly (random 1 to 10?) specific to image.X_i
    # initialize bias value w_0 specific to image.X_i

    # run scoring calculation for specific digit
    # check which number digit has highest score for and use that digit as prediction
    # if prediction is wrong:
    #   w(d)_n = w(d)_n - phi_n and w(d)_n = w(d)_n - phi_n for weights of incorrect digit predicted (where d denotes digit)
    #   w_0 = w_0 - 1
    #   w(d)_n = w(d)_n + phi_n and w(d)_n = w(d)_n + phi_n for weights of correct digit (where d denotes digit)
    #   w_0 = w_0 - 1
    return None

# When running algorithms, force machine to stop after a set number of time has passed and give stats pertaining to predictions
# e.g. accuracy % and std