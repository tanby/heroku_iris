import pickle
import numpy as np

# load previous trained model
clf2 = pickle.load(open("iris_trained_model.p", "rb"))
# array containing the labels of the output
classes = ['setosa', 'versicolor', 'virginica']

# function to do prediction
def predict_iris(slen, sw, plen, pw):
    # prepare the input as a list
    X = [slen, sw, plen, pw]

    # convert from 1D list to numpy 2D array. 1 sample = 1 row
    X = np.reshape(X, (1,-1))

    # predict the class, returning the index
    y_hat = clf2.predict(X)

    # get the label of the result based on index
    label_result = classes[y_hat[0]]  

    return label_result
