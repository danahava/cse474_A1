import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  1 / (1 + np.exp(-z))
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    #Loads the MAT object as a Dictionary
    mat = loadmat('mnist_all.mat')

    #'Constant' declaration
    TEST_DATA_SIZE = 50000
    DATA_FEATURE = 28*28

    #Initialize vectors
    train_data = np.empty([0, DATA_FEATURE])
    train_label = np.empty([0, 1])

    validation_data = np.empty([])
    validation_label = np.empty([])

    test_data = np.empty([0, DATA_FEATURE])
    test_label = np.empty([0, 1])

    for num in range(10):
        #Stack training data and label
        data = mat.get('train' + str(num))               # This is getting the training data from the matlab script (shouldn't this be 50000 NOT 10?)
        label = np.zeros( (data.shape[0], 1) )           # creates an array of zeros with rows: data.shape[0](number of rows in data) and columns: 1
        label += num                                     #""" why add to the array the numbers 1-10?"""
        train_data = np.vstack( (train_data, data) )     #"""isn't train_data empty?"""
        train_label = np.vstack( (train_label, label) )  #"""aren't these empty"""

        #Stack testing data and label
        data = mat.get('test' + str(num))                #"""But there are more than 10 test data sets""" 
        label = np.zeros( (data.shape[0], 1) )
        label += num
        test_data = np.vstack( (test_data, data) )       #"""see above"""
        test_label = np.vstack( (test_label, label) )    #"""see above"""

    #Normalize training and testing data to [0,1]
    train_data /= 255                                    #"""How is this normalizing? Should these be mode (%)"""
    test_data /= 255

    #Feature selection - remove column that has same value in all its entries
    selection = (train_data == train_data[0, :])
    selection = np.all(selection, axis=0) #axis = vertical

    for col in range(DATA_FEATURE-1, -1, -1):
        if selection[col]:
            train_data = np.delete(train_data, col, 1)

    #Split training data into training and validation
    perm = np.random.permutation(range(train_data.shape[0]))
    validation_data = train_data[ perm[TEST_DATA_SIZE:], :]
    validation_label = train_label[ perm[TEST_DATA_SIZE:], :]

    train_data = train_data[ perm[0: TEST_DATA_SIZE], :]
    train_label = train_label[ perm[0: TEST_DATA_SIZE], :]

    print("Valid: %s, %s" % (validation_data.shape, validation_label.shape))
    print("Train: %s, %s" % (train_data.shape, train_label.shape))
    print("Test: %s, %s" % (test_data.shape, test_label.shape))
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the # Directly incorporate lambda into the error function
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    #####################################################################################################################
    #data.dot(w1) will create a matrix with each row a hidden vector related to the corresponding row in the input matrix
    #So, we will compute the dot product of data.w(1), then apply the sigmoid function to each of the calulated values
    training_data = np.c_[training_data, np.ones(training_data.shape[0])] # adding a column of one to data
    a = training_data.dot(np.transpose(w1))
    z = sigmoid(a)

    ######################################################################################################################
    #z.dot(w2) will create a matrix with each row an output vector related to the corresponding row in the hidden matrix
    #So, we will compute the dot product of z.w(2), then apply the sigmoid function to each of the calculated values
    z = np.c_[z, np.ones(z.shape[0])] # adding a column of one to data
    o = z.dot(np.transpose(w2))
    y = sigmoid(o)

    #BEGIN BACK PROPOGATION:
    #####################################################################################################################
    # The first set of for loops determines the error of the weights associated with the output layer
    # Error between hidden and output
    # NOTE: lambda is directly incorporated into the calculation for the error!!!!!!!!!!!!!!!!
    delta_l = np.zeros(1, o.shape[0])
    J_2 = np.zeros(o.shape[0], shape[0]) # this should be a matrix of l*j where l is the length of the output vector and j is the length of the hidden vector 
    for l in range(0, len(truth_label)):
        for j in range(0, len(z)):
            delta_l = (training_label[l] - o[l]) * (1 - o[l]) * o[l] # these need to be looked at closer
            J_2[l][j] = -lambdaaval * delta_l[l] * z[j]
        
    #####################################################################################################################
    # The second set of for loops determines the error of the weights associated with the hidden layer
    # Error for weights between hidden and input
    # NOTE: lambda is directly incorporated into the calculation for the error!!!!!!!!!!!!!!!!
    """This still needs to be defined"""
#    J_1 = 

    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input         #for every hidden layer node we want to go down the column 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in hidden        #for every output layer node we want to go down the column 
    %     layer to unit j in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 

    #####################################################################################################################
    #A vector to hold the assigned labels for each input given in the data matrix (column vector with each row corresponding to the same input matrix row
    labels = np.empty([data.shape[0], 1])
#    print ("w1: "  + str(w1.shape))
#    print ("w2: " + str(w2.shape))
#    print ("data: " + str(data.shape))
    #####################################################################################################################
    #data.dot(w1) will create a matrix with each row a hidden vector related to the corresponding row in the input matrix
    #So, we will compute the dot product of data.w(1), then apply the sigmoid function to each of the calulated values
    data = np.c_[data, np.ones(data.shape[0])] # adding a column of one to data
    a = data.dot(np.transpose(w1))
    z = sigmoid(a)

    ######################################################################################################################
    #z.dot(w2) will create a matrix with each row an output vector related to the corresponding row in the hidden matrix
    #So, we will compute the dot product of z.w(2), then apply the sigmoid function to each of the calculated values
    z = np.c_[z, np.ones(z.shape[0])] # adding a column of one to data
    o = z.dot(np.transpose(w2))
    y = sigmoid(o)
    
    ########################################################################################################################
    #To chose which number (label) a given output vector is assigned we need to determine the maximum output value of each output
    #vector the index of that maximum value is the label assigned to the corresponding input 
    for index in range(0,y.shape[0]):
        labels[index] = np.argmax(y[index])

    return labels
    


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;

nnPredict(initial_w1, initial_w2, train_data)

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
