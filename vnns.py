
# coding: utf-8

# ## <font color=darkblue> VERSATILE NEURAL NETWORK SOLVER (VNNS)

# In[1]:

import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# In[2]:

import scipy.optimize as op
from sklearn.preprocessing import PolynomialFeatures
import timeit
import time
from IPython.display import display
import matplotlib.cm as cm
import random
import ast
import os
import sys


# In[3]:

import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='rbykgmc')
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.grid'] = 'True'
mpl.rcParams['font.size'] = 13
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['figure.figsize'] = [5.0, 4.0]


# In[4]:

class PreProcess():  
    """ Processes input data for subsequent deep learning operations. Starting from a raw .csv file, it allows to 
        normalize, add polynomial features, and partition the data into training and validation sets. 
    Parameters:
        datafile -- .csv file including feature matrix and target vector
        normalize -- boolean that allows data normalization if True (default = False)        
        polydeg -- boolean that allows feature matrix to be augmented with polynomial terms if > 1 (int, default = 1)        
        partition -- boolean that allows partitioning of the dataset into training and validation sets if True 
                    (default = False)
        fts -- fraction of training samples used for partition. Fixed at 0.7. 
    Attributes:
        original_labels_ -- list containing original class labels from the datafile
        numclass_ -- number of classes associated with the target vector (int, > 1)
        mu_ -- array containing normalization mean of feature matrix (num_features,)  
        sigma_ -- array containing normalization standard deviation of feature matrix (num_features,)
    """
        
    def __init__(self, datafile, normalize=False, polydeg=1, partition=False):       
        self.datafile = datafile
        self.normalize = normalize
        self.polydeg = polydeg
        self.partition = partition
        self.fts = 0.7
    
    def read_datafile(self, dfile=''): 
        """ Reads datafile (a csv file or alike with or without a header) and converts it to a data matrix. 
        Parameters:
            dfile -- .csv datafile (optional). If not specified self.datafile is being read. 
        Returns:
            data_array -- data matrix (num_samples, num_features+1)        
        """
        file = dfile if dfile else self.datafile
        c = pd.read_csv(file, nrows = 1).columns # read the first row
        try:
            np.array(c).astype(float) # if cols can be converted to float w/o error, that means there is no header
            data_array = pd.read_csv(file, header = None).values 
        except ValueError:            
            data_array = pd.read_csv(file).values
        return data_array
          
    def data_array_to_xy(self, data_array):
        """ Partitions the data matrix to a feature matrix and a target vector with standard labels.        
        Parameters:
            data_array -- data matrix (num_samples, num_features+1)      
        Returns:
            X -- feature matrix (num_samples, num_features)
            y -- target vector containing standard (int >= 0) class labels (number of training samples, 1)        
        """ 
        X = data_array[:,:-1].astype(float) # all cols except the last one 
        y_raw = data_array[:,-1][:,None] # last column of data array
        y = self.standardize_class_labels(y_raw)
        return X,y
    
    def standardize_class_labels(self, y_raw): 
        """ Converts class labels to int >= 0.         
        Parameters:
            y_raw -- target vector containing raw class labels (num_samples, 1)       
        Returns:
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1)        
        """
        self.original_labels_ = np.unique(y_raw).tolist()
        self.numclass_ = len(self.original_labels_)
        assert self.numclass_ >= 2
        y = y_raw
        for i in range (self.numclass_):
            A = np.where(y_raw == self.original_labels_[i])[0] # locations of elements with i'th  label in y_raw,  
            y[A] = i # assign i to those locations ==> 0,1,2, follows the alphabetical order
        y = y.astype(int)
        return y
       
    def normalize_features(self, X):
        """ Normalizes the feature matrix.     
        Parameters:
            X -- feature matrix (num_samples, num_features)      
        Returns:
            Xnorm -- normalized feature matrix where each column has a mean of ~0 and std deviation of 1 
                    (num_samples, num_features)  
        """
        m,n = X.shape # m: number of samples, n: number of features
        Xnorm = np.zeros((m,n))
        self.mu_ = np.mean(X,axis = 0)
        self.sigma_ = np.std(X,axis = 0)
        for j in range (n):
            Xnorm[:,j] = (X[:,j] - self.mu_[j])/self.sigma_[j]
        return Xnorm
    
    def add_polynomial_features(self, X): #ACKNOWLEDGMENTS: SCIKIT-LEARN
        """ Generates features for polynomial regression.
        Parameters:
            X -- feature matrix (num_samples, num_features)
        Returns:
            Xpoly -- new feature matrix augmented with polynomial features (dimension depends on the polynomial 
                     degree & num_features)              
        """
        poly = PolynomialFeatures(self.polydeg, include_bias=False)
        Xpoly = poly.fit_transform(X)
        return Xpoly
        
    def partition_training_set(self, X, y, seedno):
        """ Partitions the data set into training and validation sets.     
        Parameters:
            X -- feature matrix (num_samples, num_features) 
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1) 
            seedno -- seed for randomization of partition into training and validation sets (int)
        Returns:
            Xtr -- training feature matrix (int(self.fts*num_samples), num_features) 
            ytr -- training target vector (int(self.fts*num_samples), 1)
            Xval -- validation feature matrix (1-int(fts*num_samples), num_features) 
            yval -- validation target vector (1-int(fts*num_samples), 1)
        """
        data_arr = np.column_stack((X,y))
        m,n = X.shape
        np.random.seed(seedno)
        np.random.shuffle(data_arr)                   
        train_set = data_arr[:int(m*self.fts)+1] 
        valid_set = data_arr[int(m*self.fts)+1:]
        Xtr = train_set[:,0:n]
        ytr = train_set[:,n][:,None].astype(int)
        Xval = valid_set[:,0:n]
        yval = valid_set[:,n][:,None].astype(int)
        if len(np.unique(ytr)) < self.numclass_ or len(np.unique(yval)) < self.numclass_:
            print("training and/or validation set has less than the orginal number of classes")
            assert(len(np.unique(ytr)) == self.numclass_ and len(np.unique(yval)) == self.numclass_)
            # stop if partitioned sets do not include at least one sample from each class
        return Xtr, ytr, Xval, yval
    
    def process_training_set(self, seedno=0):
        """ Combines reading input file, label standardization, feature normalization, polynomial feature addition, 
            and partitioning of input training data. Returned vector/matrix dimensions depends on whether there is 
            polynomial regression(which would increase the number of features) and partition (which would change the 
            number of training and validation samples.)
        Parameters:
            seedno -- seed for randomization of partition into training and validation sets (int)         
        Returns:
            X -- pre-processed training feature matrix (training final num_samples, final num_features) 
            y -- pre-processed training target vector (training final num_samples, 1)
            Xval -- pre-processed validation feature matrix (validation final num_samples, final num_features) 
            yval -- pre-processed validation target vector (validation final num_samples, 1)
        """  
        data_array = self.read_datafile()
        X,y = self.data_array_to_xy(data_array)
        Xval = None # overwritten below if self.partition = True
        yval = None
        if self.normalize == True:
            X = self.normalize_features(X) 
        if self.polydeg > 1:
            X = self.add_polynomial_features(X) # In case both normalization and polynomial regression were 
                                                # set to True, normalization is implemented first.
        if self.partition == True:
            X,y,Xval,yval = self.partition_training_set(X,y,seedno) # X, y = Xtr, ytr
        return X, y, Xval, yval

    def process_test_set(self, test_file,mu_tr='', sigma_tr=''):
        """ Combines reading, label standardization, feature normalization, and polynomial feature addition for the 
            test set.  
        Parameters:
            test_file -- a separate data file never used by the network during training
            mu_tr -- array containing normalization mean from the training set, optional in case training set is 
                     actually not normalized(num_features,)  
            sigma_tr -- array containing normalization standard deviation from the training set, optional in case 
                        training set is actually not normalized(num_features,)  
        Returns:
            Xtest -- pre-processed test feature matrix (test num_samples, final num_features) 
            ytest -- pre-processed test target vector (test num_samples, 1)
        """  
        test_array = self.read_datafile(test_file)
        Xtest,ytest = self.data_array_to_xy(test_array)
        mtest,ntest = Xtest.shape
        if self.normalize == True:
            for j in range (ntest):                
                Xtest[:,j] = (Xtest[:,j] - mu_tr[j])/sigma_tr[j]
        if self.polydeg > 1:
            Xtest = self.add_polynomial_features(Xtest) 
        return Xtest, ytest


# In[5]:

class DeepSolve():    
    """ Computes parameters for neural networks with arbitrary number of hidden layers and units in each layer. 
        Makes predictions based on calculated parameters and reports their accuracy. Accommodates multi-class 
        classification. A custom gradient descent method (with "auto" learning-rate option) as well as 
        scipy.optimize.minimize methods are available to solve the parameters. 
    Parameters:
        method -- batch solver based on either a custom gradient descent implementation ('GD' which is the default) 
                  or other built-in algorithm implementations under scipy.optimize.minimize (such as 'TNC','CG',
                  'SLSQP','BFGS','L-BFGS-B') 
                  See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
        initialize -- type of initialization for network parameters. "zeros" results in initial parameters being all 
                      zero. "deep" (default) results in He initialization
        learning_rate -- learning rate for gradient descent. Effective only if method = 'GD'. Can be either a float 
                         > 0 or 'auto' (default) which would allow 'GD' to auto-select the learning rate
        lamda -- regularization parameter (float >= 0, default = 0)
        maxiter -- max number of iterations (int >= 1, default = 1000)
        hidden units -- list showing number of units in each hidden layer. As an example, hidden_units = [20,15,10] 
                        refers to a network with three hidden layers and 20 units in the first layer and so on. 
                        hidden_units = [] (default) means there are no hidden units (in which case the operation is 
                        simple logistic regression)
        L -- total number of layers in the network, equals to number of hidden layers + 2 to account for input and 
             output layers (value is tied to parameter hidden_units)
        minlr -- minimum allowable learning rate if parameter learning_rate is set to 'auto'. Effective only if 
                 method = 'GD'. Fixed at 1e-8
        crit -- convergence criterion for 'GD' (so effective only if method = 'GD'.) 'GD' converges when cost decrease 
                between two subsequent iterations divided by the actual cost becomes less than the parameter crit. 
                Fixed at 2e-5
    Attributes:
        Jh_ -- array containing the cost value at each iteration  including the initial cost (numiter+1,)
        message_ -- solver result, e.g., "Converged" (string)
        niter_ -- number of iterations (int > 0)
        lrfinal_ -- auto-selected value of learning rate for method = 'GD' and learning_rate = 'auto' (float)
        timetofit_ -- execution time taken by solver (sec)
    """
    
    def __init__(self, method='GD', initialize='deep', learning_rate='auto', lamda=0, maxiter=1000, hidden_units=[]):
        self.hidden_units = hidden_units
        self.method = method
        self.lamda = lamda
        self.maxiter = maxiter
        self.initialize = initialize
        self.learning_rate = learning_rate 
        self.L = len(hidden_units)+2       
        self.minlr = 1e-8
        self.crit = 2e-5
        
    def relu(self, x):
        """ Rectifying linear unit activation function.
        Parameters:
            x -- array (any size)
        Returns:
            r -- relu(x) = x if x > 0 else 0
        """
        r =  x * (x > 0)
        return r

    def gradrelu(self, x): 
        """ Gradient of rectifying linear unit activation function.
        Parameters:
            x -- array (any size)
        Returns:
            gr -- gradient of relu(x) = 1 if x > 0 else 0
        """
        gr = (x > 0) * 1
        return gr

    def sigmoid(self, x):
        """ Sigmoid activation function.
        Parameters:
            x -- array (any size)
        Returns:
            s -- sigmoid(x)
        """
        s = 1/(1 + np.exp(-x))
        return s

    def add_bias(self, X):
        """ Inserts a vector of ones as the first column to the designated matrix.
        Parameters:
            X -- feature matrix (num_samples,num_features)
        Returns:
            X1 -- feature matrix augmented with a bias vector of ones as the first column (num_samples,num_features+1)
        """
        X1 = np.insert(X,0,1,axis=1)
        return X1

    def convert_to_one_hot(self, y): 
        """Converts target vector to one-hot matrix if number of classes > 2.
        Parameters:
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1) 
        Returns:
            ymc -- y itself (num_samples, 1) or one-hot matrix (num_samples,num_classes)
        """
        ymc = y
        classes = np.unique(y)
        numc = len(classes)
        if numc > 2:
            ymc = (y==classes[0])*1
            for i in range (1,numc):
                ymc = np.column_stack((ymc,(y==classes[i])*1)) 
        return ymc

    def set_layer_sizes(self, X1, y):
        """ Sets layer sizes including input, hidden, and output units.
        Parameters:
            X1 -- feature matrix augmented with a bias vector of ones as the first column (num_samples,num_features+1)                 
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1) 
        Returns:
            layer_sizes -- list holding number of units in each layer
        """
        num_input_units = X1.shape[1]-1 # X1 is preferred to X as parameter to speed-up parameter fitting
        classes = np.unique(y)
        num_out_units = len(classes) if len(classes) > 2 else 1 # if there are only two classes, there is no need for 
                                                                # two ouput units whose outputs would be complementary
                                                                # with probabilities p and 1-p
        layer_sizes = [num_input_units] + self.hidden_units + [num_out_units] 
        return layer_sizes

    def initialize_parameters(self, layer_sizes):
        """ Initializes parameters theta (i.e., bias vector concatenated with weight matrix.) 
        Parameters:
            layer_sizes -- list holding number of units in each layer                 
        Returns:
            theta_init -- dictionary holding initialized network parameters 
                          keys: int > 0, total number of keys: num_layers - 1
                          dimensions for each key: (num_units_next, num_units_prev+1)
        """ 
        theta_init = {} 
        np.random.seed(3)
        for l in range(1, self.L):
            if self.initialize == "deep":  
                w = np.random.randn(layer_sizes[l], layer_sizes[l-1])/np.sqrt(layer_sizes[l-1]) # He initialization
            if self.initialize == "zeros":
                w = np.zeros((layer_sizes[l], layer_sizes[l-1]))
            b = np.zeros((layer_sizes[l], 1)) # bias always initialized to zero    
            theta_init[l] = np.column_stack((b,w))
        return theta_init

    def fwd_prop(self, theta, X1):
        """ Calculates pre-activation and activation matrices for each layer of the network through a forward 
            propagation from input to output. Relu activation is used at all layers except the output where sigmoid 
            activation is used.
        Parameters:
            theta -- dictionary holding network parameters
                     keys: int > 0, total number of keys: num_layers - 1
                     dimensions for each key: (num_units_next, num_units_prev+1)
            X1 -- feature matrix augmented with a bias vector of ones as the first column (num_samples,num_features+1)                
        Returns:
            AL -- activation matrix at the output (1 or num_classes, num_samples)
            A1 -- dictionary holding activation matrices with bias added as the first row
                  keys: int > 0, total number of keys: num_layers - 1
                  dimensions for each key: (num_units+1, num_samples)        
        """ 
        A = {}
        A1 = {}
        Z = {}    
        A1[1] = X1.T
        k = 2
        while k <= self.L:
            Z[k] = np.dot(theta[k-1], A1[k-1])
            if k < self.L:               
                A[k] = self.relu(Z[k])
            elif k == self.L:
                A[k] = self.sigmoid(Z[k])
            if k < self.L:
                A1[k] = np.insert(A[k],0,1,axis=0) 
            k = k+1
        AL = A[self.L]
        return Z,AL,A1
    
    def back_prop(self, theta, y, AL, Z):
        """ Calculates the error matrices for each layer of the network except the input through a backward 
            propagation from output to input. 
        Parameters:
            theta -- dictionary holding network parameters
                     keys: int > 0, total number of keys: num_layers - 1
                     dimensions for each key: (num_units_next, num_units_prev+1)
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1) 
            AL -- activation matrix at the output (1 or num_classes, num_samples)
            Z -- dictionary holding pre-activation matrices
                 keys: int > 1, total number of keys: num_layers - 1
                 dimensions for each key: (num_units, num_samples)             
        Returns:            
            dZ -- dictionary holding error matrices associated with each layer except the input
                  keys: int > 1, total number of keys: num_layers - 1
                  dimensions for each key: (num_units, num_samples)
        """
        dZ = {}
        l = self.L
        ymc = self.convert_to_one_hot(y)  
        while l >=2 :
            if l == self.L:
                dZ[l] = AL - ymc.T 
            elif l < self.L:
                theta_r_l = theta[l][:,1:] #strip out the bias vector from the first column
                theta_r_T_l = theta_r_l.T
                dA_l = np.dot(theta_r_T_l,dZ[l+1])
                grZ_l = self.gradrelu(Z[l])                
                dZ[l] = dA_l*grZ_l
            l -= 1
        return dZ
        
    def cost_nn(self, theta, X1, y):
        """ Calculates cross-entropy cost including the regularization term.
        Parameters:
            theta -- dictionary holding network parameters
                     keys: int > 0, total number of keys: num_layers - 1
                     dimensions for each key: (num_units_next, num_units_prev+1)
            X1 -- feature matrix augmented with a bias vector of ones as the first column (num_samples,num_features+1)
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1) 
        Returns:
            cost -- cost based on cross-entropy loss (-ylog(AL)-(1-y)log(1-AL))) + regularization term (float)
        """
        eps = 1e-323 # prevents logarithm of zero in Jdiag below
        m = len(y)
        ymc = self.convert_to_one_hot(y)
        Z, AL, A1 = self.fwd_prop(theta,X1)
        Jdiag =-np.einsum('ij,ji->i',ymc.T,np.log10(AL.T+eps))/m-np.einsum('ij,ji->i',(1-ymc).T,np.log10(1-AL.T+eps))/m
        Jnoreg = np.sum(Jdiag) # Cost without regularization 
        # einsum formulation above replaces the for loop below in case of more than two classes (and also works for 
        # two classes):  
        # Jnoreg = 0
        # for k in range (0,len(np.unique(y))):
        #    yy = ymc[:,k][:,None]      
        #    hh = AL.T[:,k][:,None]
        #    J_lam0 = (-np.dot(yy.T,np.log10(hh+eps))-np.dot((1-yy).T,np.log10(1-hh+eps)))/m
        #    Jnoreg = Jnoreg + J_lam0[0][0]
        Jreg = 0
        for i in range (1,self.L):
            theta_r = theta[i][:,1:]
            Jreg += (self.lamda/(2*m))*sum(sum(theta_r**2)) # Regularization term
        cost = Jnoreg + Jreg
        return cost  

    def grad_nn(self, theta, X1, y):
        """ Calculates the gradient of cost (including the regularization term) with respect to theta.
        Parameters:
            theta -- dictionary holding network parameters
                     keys: int > 0, total number of keys: num_layers - 1
                     dimensions for each key: (num_units_next, num_units_prev+1)
            X1 -- feature matrix augmented with a bias vector of ones as the first column (num_samples,num_features+1)
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1) 
        Returns:
            grad -- dictionary holding the gradient of cost for each theta
                    keys: int > 0, total number of keys: num_layers - 1
                    dimensions for each key: (num_units_next, num_units_prev+1)
        """
        m = y.shape[0]        
        grad = {}            
        Z,AL,A1 = self.fwd_prop(theta,X1)
        dZ = self.back_prop(theta,y,AL,Z)
        l = self.L
        while l >=2 :
            grad_noreg = np.dot(dZ[l],A1[l-1].T)/m # unregularized term
            grad_reg = np.array(theta[l-1]*(self.lamda/m)) # regularized term
            grad_reg[:,0] = 0 # don't regularize the bias term
            grad[l-1] = grad_noreg + grad_reg
            l -= 1
        return grad
    
    def fit_gdnn(self, X, y):
        """ Finds network parameters theta by gradient descent. Learning rate can either be set by the user or be 
            auto-selected.
        Parameters:
            X -- feature matrix (num_samples,num_features)
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1) 
        Returns:
            theta -- dictionary holding network parameters
                     keys: int > 0, total number of keys: num_layers - 1
                     dimensions for each key: (num_units_next, num_units_prev+1)
        """           
        X1 = self.add_bias(X)
        layer_sizes = self.set_layer_sizes(X1,y)
        theta_init = self.initialize_parameters(layer_sizes) 
        cost_init = self.cost_nn(theta_init,X1,y) 
        theta = theta_init
        self.Jh_ = cost_init
        alp = 10 if self.learning_rate == 'auto' else self.learning_rate # initial learning rate in "auto" mode
        i = 0
        while i < self.maxiter and alp > self.minlr:        
            i = i + 1 
            grad = self.grad_nn(theta,X1,y)
            for j in range (1,len(layer_sizes)):
                theta[j] -= alp*grad[j] # update theta 
            self.Jh_ = np.append(self.Jh_,self.cost_nn(theta,X1,y)) # storing cost at each step tends to increase 
                                                                    # execution time significantly
            if (self.Jh_[i-1] - self.Jh_[i])<= 0:
                if self.learning_rate == 'auto':
                    alp = alp*0.3 # in "auto" mode, reduce learning rate by 0.3 until cost decreases monotonically
                    theta_init = self.initialize_parameters(layer_sizes) 
                    cost_init = self.cost_nn(theta_init,X1,y) 
                    theta = theta_init # re-initialization is more effective than continuing from where it diverged
                    self.Jh_ = cost_init          
                    i = 0    
            elif 0 <(self.Jh_[i-1] - self.Jh_[i])/ self.Jh_[i] < self.crit: # stop when delta cost/cost < convergence 
                                                                            # threshold
                self.message_ = "Converged"
                break
        self.niter_ = i
        self.lrfinal_ = alp 
        if self.niter_ == self.maxiter:
            self.message_ = "Max number of iterations reached" 
        elif self.lrfinal_ < self.minlr:
            self.message_ = "No convergence (learning rate too low)" 
        return theta
    
    def unroll(self, dict):
        """ Maps dictionary values into a column vector.
        Parameters:
            dict -- a dictionary where each value can be a matrix
        Returns:
            dict_unrolled -- vector storing dictionary values in unrolled form (num_dict_elements,1) 
        """
        dict_unrolled = np.array([])
        keys = list(dict.keys())        
        for k in keys:
            t = np.array(dict[k])
            dict_unrolled = np.append(dict_unrolled,np.ravel(t))        
        dict_unrolled = dict_unrolled[:,None]
        return dict_unrolled
      
    def dict_dims(self, dict):
        """ Holds dimensions of matrices that are values to a dictionary.
        Parameters:
            dict --  a dictionary where each value can be a matrix
        Returns:
            dims -- a matrix with each column storing the shape of a matrix associated with the values of 
                    a dictionary (2, num_keys) 
        """
        keys = list(dict.keys())
        dims = np.array(dict[keys[0]].shape)
        for i in range (1, len(dict.keys())):
            dims = np.row_stack((dims,np.array(dict[keys[i]].shape)))
        dims = dims.T.reshape((2,len(dict.keys())))
        return dims   
    
    def roll(self, dict_unrolled,dims):
        """ Restores original dictionary.
        Parameters:
            dict_unrolled -- vector storing authentic dictionary values in unrolled form (num_dict_elements,1)
            dims -- a matrix with each column storing the shape of a matrix associated with the values of 
                    an authentic dictionary (2, num_keys)
        Returns:
            dict_authentic --  authentic dictionary with keys: int > 0  
        """
        dict_authentic = {}
        len_prev = 0
        for i in range(0,dims.shape[1]):
            num_elements_i = dims[0,i]*dims[1,i]
            w = dict_unrolled[len_prev : len_prev + num_elements_i] # slice dict_unr vector to extract elements 
            dict_authentic[i+1] = w.reshape((dims[0,i],dims[1,i])) # reshape those according to dims
            len_prev = len_prev + num_elements_i
        return dict_authentic
        
    def fit_scipynn(self, X, y):
        """ Finds network parameters theta by using scipy's built-in minimization methods. Local functions defined 
            within the main function allow to accommodate the use of unrolled network parameters and gradients as 
            required by scipy.optimize.minimize.
        Parameters:
            X -- feature matrix (num_samples,num_features)
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1) 
        Returns:
            theta -- dictionary holding network parameters
                     keys: int > 0, total number of keys: num_layers - 1
                     dimensions for each key: (num_units_next, num_units_prev+1)
        """ 
        X1 = self.add_bias(X)
        layer_sizes = self.set_layer_sizes(X1,y)
        theta_init = self.initialize_parameters(layer_sizes) 
        cost_init = self.cost_nn(theta_init,X1,y) 
        self.Jh_ = cost_init          
        theta_init_unrolled = self.unroll(theta_init)
        theta_dims = self.dict_dims(theta_init)
        def cost_nn_red(theta_unrolled, X1, y):
            """Similar to cost_nn method except that theta_unrolled replaces theta as parameter """
            theta = self.roll(theta_unrolled,theta_dims)
            cost = self.cost_nn(theta,X1,y)
            return cost
        def grad_nn_red(theta_unrolled, X1, y):
            """Similar to grad_nn method except that theta_unrolled replaces theta as parameter and grad_unrolled 
               is returned instead of grad"""
            theta = self.roll(theta_unrolled,theta_dims)
            grad = self.grad_nn(theta, X1, y)
            grad_unrolled = self.unroll(grad).T[0]
            return grad_unrolled
        def track_cost(theta_unrolled):
            """Stores cost values at each iteration"""
            self.Jh_ = np.append(self.Jh_,cost_nn_red(theta_unrolled,X1,y))
            return self       
        fmin = op.minimize(fun = cost_nn_red, x0 = theta_init_unrolled, args = (X1, y), callback = track_cost,
                           method = self.method, jac = grad_nn_red, options={'maxiter': self.maxiter})       
        self.niter_ = fmin.nit
        self.message_ = fmin.message     
        theta = self.roll(fmin.x,theta_dims)     
        return theta
       
    def fit_nn(self, X, y):
        """ Finds network parameters theta either by gradient descent or using scipy's built-in minimization methods 
            Also keeps track of execution time.
        Parameters:
            X -- feature matrix (num_samples,num_features)
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1) 
        Returns:
            theta -- dictionary holding network parameters
                     keys: int > 0, total number of keys: num_layers - 1
                     dimensions for each key: (num_units_next, num_units_prev+1)
        """ 
        tic = timeit.default_timer()
        theta = self.fit_gdnn(X,y) if self.method == "GD" else self.fit_scipynn(X,y)
        toc = timeit.default_timer()        
        self.timetofit_ = np.round(toc-tic,6)
        return theta
            
    def predict(self, theta, X):
        """ Predicts labels based on network parameters theta.
        Parameters:
            theta -- dictionary holding network parameters
                     keys: int > 0, total number of keys: num_layers - 1
                     dimensions for each key: (num_units_next, num_units_prev+1)
            X -- feature matrix (num_samples,num_features)
        Returns:
            y_hat -- array of predicted output in standard form of int >= 0 (num_samples,)
        """
        keys = list(theta.keys())
        num_row_last_theta = theta[max(keys)].shape[0]
        num_class = 2 if num_row_last_theta == 1 else num_row_last_theta
        X1 = self.add_bias(X)
        Z,AL,A1= self.fwd_prop(theta,X1)
        if num_class == 2:
            pred = np.where(AL.T >= 0.5, 1, 0)
        elif num_class > 2:                       
            pred = AL.T.argmax(axis=1).reshape((len(X),1))
        y_hat = np.squeeze(pred)
        return y_hat
    
    def score(self, y_hat, y):
        """ Measures the accuracy of predictions by comparing them with the true labels.
        Parameters:
            y_hat -- array of predicted output containing standard (int >= 0) class labels (num_samples,)
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1) 
        Returns:
            accuracy -- number of accurately predicted labels over the number of all predictions (float, min:0, max:1)
        """
        ind_where_y_equals_yhat = np.where(y_hat == np.ravel(y))[0]
        accuracy = len(ind_where_y_equals_yhat)/len(y)
        return accuracy 


# In[6]:

class DeepCombine():    
    """ Combines two classes, PreProcess() and DeepSolve(), and complements them with plotting and report generating 
        functions. It absorbs all intermediate steps to generate the results (weights, scores..) as directly as 
        possible starting from the raw dataset.
    Parameters:
        datafile -- .csv file including feature matrix and target vector
        normalize -- boolean that allows data normalization if True (default = False)        
        polydeg -- boolean that allows feature matrix to be augmented with polynomial terms if > 1 (int, default = 1)        
        partition -- boolean that allows partitioning of the dataset into training and validation sets if True 
                    (default = False)
        method -- batch solver based on either a custom gradient descent implementation ('GD' which is the default) 
                  or other built-in algorithm implementations under scipy.optimize.minimize (such as 'TNC','CG',
                  'SLSQP','BFGS','L-BFGS-B') 
                  See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
        initialize -- type of initialization for network parameters. "zeros" results in initial parameters being all 
                      zero. "deep" (default) results in He initialization
        learning_rate -- learning rate for gradient descent. Effective only if method = 'GD'. Can be either a float 
                         > 0 or 'auto' (default) which would allow 'GD' to auto-select the learning rate
        lamda -- regularization parameter (float >= 0, default = 0)
        maxiter -- max number of iterations (int >= 1, default = 1000)
        hidden units -- list showing number of units in each hidden layer. As an example, hidden_units = [20,15,10] 
                        refers to a network with three hidden layers and 20 units in the first layer and so on. 
                        hidden_units = [] (default) means there are no hidden units (in which case the operation is 
                        simple logistic regression)
        pp -- instance of Class PreProcess()
        ds -- instance of Class DeepSolve()
        colors -- list of colors enabling to color match the scatter and decision boundary plots 
    """
    def __init__(self, datafile, normalize=False, polydeg=1, partition=False, 
                 method='GD', initialize = 'deep', learning_rate='auto', lamda=0, maxiter=10000, hidden_units=[]):
        self.pp = PreProcess(datafile,normalize, polydeg, partition)        
        self.ds = DeepSolve(method, initialize, learning_rate, lamda, maxiter, hidden_units)
        self.colors = ['blue','orange','red','lime','purple','yellow','cyan','magenta','black','purple']

    def plt_input(self,X,y):
        """ Generates a 2D plot mapping color-coded class labels to the first two features X1 and X2.
        Parameters:
            X -- feature matrix (num_samples,num_features)
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1) 
        """
        for i in range(self.pp.numclass_): 
            z = np.where(y==i)[0]
            plt.scatter(X[z,0],X[z,1],facecolor = self.colors[i], linewidth = 1, label=self.pp.original_labels_[i],
                        edgecolor = "black",s = 40)
        plt.legend (loc='best')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('training dataset')
                     
    def plot_J(self): 
        """ Plots normalized cost function (J/J_init) vs iterations.
        """
        iter = range(1, self.ds.niter_+2) # +2: +1 due to starting w/1, other +1 due to inclusion of Jh[0]
                                          # start from 1 to accommodate x scale in log
        norm_cost_max = self.ds.Jh_[0]/self.ds.Jh_[-1]
        norm_cost = self.ds.Jh_/self.ds.Jh_[0]       
        if norm_cost_max > 10 and self.ds.niter_>10:
            plt.loglog(iter, norm_cost ,'-', linewidth = 3) # use log scale if range > 1 decade
        elif norm_cost_max <= 10 and self.ds.niter_> 10:
            plt.semilogx(iter, norm_cost ,'-', linewidth = 3)
        elif norm_cost_max > 10 and self.ds.niter_ <= 10:
            plt.semilogy(iter, norm_cost ,'-', linewidth = 3)
        elif  norm_cost_max <= 10 and self.ds.niter_ <= 10:
            plt.plot(iter, norm_cost ,'-', linewidth = 3)
        plt.xlim(xmin=1)
        plt.xlabel('Number of Iterations') #in fact this is numiter + 1
        plt.ylabel('Normalized cost') # Jh/Jh[0]    
        plt.grid(True,which='both')
        plt.title('Norm. cost vs num. iterations')
        plt.show()
            
    def plt_output(self, X, y, theta, y_hat, tr=True):  
        """ Generates a 2D contour plot (for the first two features X1 and X2) identifying regions belonging to 
            different classes. It also overlays input scatter plot on top of the contour plot and shows erroneous 
            predictions if any. 
        Parameters:
            X -- feature matrix (num_samples,num_features)
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1)             
            theta -- dictionary holding network parameters
                     keys: int > 0, total number of keys: num_layers - 1
                     dimensions for each key: (num_units_next, num_units_prev+1)
            y_hat -- array of predicted output in standard form of int >= 0 (num_samples,)
            tr -- boolean indicating the nature of dataset (default = True)
        """
        ### 1. Determine axis boundaries and step size 
        numstep = 250
        extfactor = 0.1 
        extra1 = (np.max(X[:,0]) - np.min(X[:,0]))*extfactor
        extra2 = (np.max(X[:,1]) - np.min(X[:,1]))*extfactor
        x1 = (np.arange(np.min(X[:,0]) - extra1, np.max(X[:,0]) + extra1, 
                        (np.max(X[:,0])-np.min(X[:,0])+2*extra1)/numstep))
        x2 = (np.arange(np.min(X[:,1]) - extra2, np.max(X[:,1]) + extra2, 
                        (np.max(X[:,1])-np.min(X[:,1])+2*extra2)/numstep))
        ### 2. Predict the output for each (x1[i], x2[j]) coordinate       
        pred = np.zeros((len(x1), len(x2)))
        for i in range (0,len(x1)):    
            for j in range (0,len(x2)):
                xx = np.array([x1[i],x2[j]]).reshape((1,2)) # select the coordinate on the 2D map
                xxx = self.pp.add_polynomial_features(xx) # add polynomial features if self.pp.polydeg > 1
                if theta[1].shape[1] != xxx.shape[1]+1:
                    print ("\nnumber of input features > 2 is not allowed w/o polynomial regression\n")
                assert theta[1].shape[1] == xxx.shape[1]+1
                pred[i,j]  = self.ds.predict(theta,xxx) # for each coordinate, predict the output based on theta   
        ### 3. Contour setup           
        num_classes = len(np.unique(y))
        levels = np.linspace(0,num_classes-1,num_classes+1)
        plt.contour(x1, x2, pred.T, levels = levels, linewidths=0.5, colors='k') # plots the decision boundary
        cont = plt.contourf(x1, x2, pred.T, levels = levels,colors = self.colors,alpha=0.4) # color different regions 
        ### 4. Overlay input scatter plot and show errors
        self.plt_input(X,y) 
        error_index = np.where(y_hat!= y.T)[1]
        plt.scatter(X[error_index,0],X[error_index,1],marker='o', facecolor='None', edgecolor='black', s=40, 
                    linewidth=2, label='errors') # show errors
        plt.legend(bbox_to_anchor=(1.3,1.03))
        plt.title('decision boundary on the tr set') if tr==True else plt.title('decision boundary on the test set')  
        plt.show() 
    
    def report(self, X, y, yval='', score_tr='', score_val='', cost_val=''):
        """ Generates a report summarizing the dataset features, solver features, and results.
        Parameters:
            X -- feature matrix (num_samples,num_features)
            y -- target vector containing standard (int >= 0) class labels (num_samples, 1) 
            yval -- validation target vector (1-int(fts*num_samples), 1)
            score_tr -- number of accurately predicted labels over the number of all predictions on the training set
                        (float, min:0, max:1)
            score_val -- number of accurately predicted labels over the number of all predictions on the validation set
                        (float, min:0, max:1)
            cost_val -- validation cost based on cross-entropy loss (-ylog(AL)-(1-y)log(1-AL))) + regularization term 
                        (float)
        """  
        ### data
        print ("\ndata:")
        print("\tfile:",self.pp.datafile)
        m,n = X.shape
        print ("\tm =", m, "training examples")
        print ("\tn =", n, "features")
        print("\toriginal classes:", self.pp.original_labels_)
        for i in range(self.pp.numclass_):
            z = np.where(y==i)[0]
            print('\tnumber of samples in class '+str(i)+" =", len(z)) 
        ### solver          
        print ("solver:")
        print ("\tfeature normalization: yes") if self.pp.normalize == True else print ("\tfeature normalization: no")            
        if self.pp.polydeg > 1:
            print ("\tpolynomial regression: yes (deg = ", str(self.pp.polydeg)+")")
        elif self.pp.polydeg == 1:
            print ("\tpolynomial regression: no")
        X1 = self.ds.add_bias(X)
        print ("\tneural network config:", self.ds.set_layer_sizes(X1,y))
        if self.ds.lamda > 0:
            print ("\tregularization: yes (lambda =",str(self.ds.lamda)+")")
        elif self.ds.lamda == 0:
            print ("\tregularization: no")
        print ("\tmethod:", self.ds.method)
        if self.ds.method == 'GD':
            print ("\tlearning rate =", round(self.ds.lrfinal_,8))  
        ### results    
        print ("output:")
        print ("\t* "+str(self.ds.message_))
        if self.ds.niter_ >= 1:            
            print("\tinitial cost =", self.ds.Jh_[0])
            print ("\tfinal cost =", self.ds.Jh_[-1])     
            print ("\tnumber of iterations =", self.ds.niter_)
            print("\taccuracy on the training set:",np.round(score_tr,3))
            print ("\texecution time: ",self.ds.timetofit_," sec")
            if self.pp.partition == True:
                    print ("validation:")   
                    for i in range(len(np.unique(yval))):
                        z = np.where(yval==i)[0]
                        print('\tnumber of samples in class '+str(i)+" =", len(z)) 
                    print ("\tcost =", cost_val)            
                    print("\taccuracy on the validation set:",np.round(score_val,3))
                
    def combine(self, seedno=0, plot_input=False, plot_J=False, plot_output=False, report_summary=False): 
        """ Combines the functions from classes PreProcess(), DeepSolve(), and plotting/reporting functions from class 
            DeepCombine() to pre-process input data, find parameters, optionally plot input/cost/decision boundary, and 
            generate a summary report.
        Parameters:
            seedno -- seed for randomization of training and validation set partition (int) 
            plot_input -- boolean that allows the dataset to be plotted if True (default=False) 
            plot_J -- boolean that allows the cost vs iterations to be plotted if True (default=False) 
            plot_output -- boolean that allows the decision boundary to be plotted if True (default=False) 
            report_summary -- boolean that allows the summary report to be generated if True (default=False) 
        Returns:
            results -- dictionary storing network parameters theta ('theta'), training score ('score_tr'), 
                       validation score (score_val'), number of iterations ('numiter'), final cost ('finalcost'), 
                       and execution time ('time')            
        """
        X,y,Xval,yval = self.pp.process_training_set(seedno)        
        if plot_input == True:    
            self.plt_input(X,y)
            plt.show();
        theta = self.ds.fit_nn(X,y)    
        if self.ds.niter_  >= 1:      
            pred_tr  = self.ds.predict(theta,X)
            score_tr = self.ds.score(pred_tr,y)       
            if self.pp.partition == True:
                pred_val = self.ds.predict(theta,Xval)
                score_val = self.ds.score(pred_val,yval)
                Xval1 = self.ds.add_bias(Xval)
                cost_val = self.ds.cost_nn(theta, Xval1, yval)
            else:
                score_val = None
                cost_val = None            
            results = {'theta':theta, 'score_tr':score_tr, 'score_val':score_val, 'numiter':self.ds.niter_, 
                       'finalcost': self.ds.Jh_[-1], 'timetofit': self.ds.timetofit_}            
            if plot_J == True:
                self.plot_J()
                plt.close()           
            if report_summary == True:
                self.report(X,y,yval,score_tr, score_val,cost_val)        
            if plot_output == True:
                self.plt_output(X,y,theta,pred_tr)                
                plt.close()           
        elif self.ds.niter_ < 1:
            score_tr  = None
            score_val = None
            cost_val = None
            results = None                      
        return results 


# In[7]:

class DeepLearnAuto():    
    """ Augments the capabilities of class DeepCombine() by automated tuning of polynomial degree, hidden-unit 
        configuration, and regularization parameter lamda. In addition, predicts classes on the test set and calculates
        the test score.
    Parameters:
        datafile -- .csv file including feature matrix and target vector
        normalize -- boolean that allows data normalization if True (default = False)        
        polydeg -- boolean that allows feature matrix to be augmented with polynomial terms if > 1 (int, default = 1)        
        method -- batch solver based on either a custom gradient descent implementation ('GD' which is the default) 
                  or other built-in algorithm implementations under scipy.optimize.minimize (such as 'TNC','CG',
                  'SLSQP','BFGS','L-BFGS-B') 
                  See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.
        initialize -- type of initialization for network parameters. "zeros" results in initial parameters being all 
                      zero. "deep" (default) results in He initialization
        learning_rate -- learning rate for gradient descent. Effective only if method = 'GD'. Can be either a float 
                         > 0 or 'auto' (default) which would allow 'GD' to auto-select the learning rate
        lamda -- regularization parameter (float >= 0, default = 0)
        maxiter -- max number of iterations (int >= 1, default = 1000)
        hidden units -- list showing number of units in each hidden layer. As an example, hidden_units = [20,15,10] 
                        refers to a network with three hidden layers and 20 units in the first layer and so on. 
                        hidden_units = [] (default) means there are no hidden units (in which case the operation is 
                        simple logistic regression)        
        num_hidden_layers -- number of hidden layers (with equal number of hidden units) for selection of optimal 
                             hidden unit configuration in auto mode (int > 0, default = 1)
        total_units_max -- total number of hidden units for selection of optimal hidden unit configuration in auto
                           mode. As an example, if num_hidden_layers = 2 and total_units_max = 10, the candidate hidden 
                           layers for two classes are [] (which is the reference), [3,3], and [4,4]. 
                           (int > 0, default=10)        
        plot_input -- boolean that allows the dataset to be plotted if True (default=False)
        plot_J -- boolean that allows the normalized cost vs iterations to be plotted if True (default=False)
        report_summary -- boolean that allows the summary report to be generated if True (default=False)
        plot_output -- boolean that allows the decision boundary to be plotted if True (default=False)
        plot_test -- boolean that allows the decision boundary to be plotted for the test set if True (default=False)
        plot_poly -- boolean that allows training score vs polynomial degree to be plotted if True (default=False) 
        plot_hidden -- boolean that allows training score vs hidden units config to be plotted if True (default=False)
        plot_lam -- boolean that allows training and validation scores vs lamda to be plotted if True
        polydeg_max -- max candidate polynomial degree to consider while selecting the optimal degree in auto mode 
                      (int >=1, fixed at 5)
        lamda_numdecade -- number of decades for optimal lamda search (int >= 1, fixed at 3)
        totalseed -- number of times dataset is randomly partitioned based on different seeds (int >= 1, fixed at 10)
        topscore -- limit ("good enough") score to stop the search for polydeg or hidden units (float, fixed at 0.95)
        deltascore -- minimum training score improvement needed to continue the search for a higher polynomial degree 
                      or total number of hidden units. The idea is to stop searching for more complex models if the 
                      training score is already saturated (float, fixed at 0.05)         
        self.valscoredelta -- during the optimal lamda search, this is the minimum validation score improvement needed 
                              to continue the search for a lower lamda. The idea is to go to lower lamda values only if 
                              validation score is going to improve. Otherwise we prefer to keeplamda high (float, fixed
                              at 0.005)
        self.deltatrscorevslam0 -- during the optimal lamda search, this is the maximum allowable difference between 
                                   the training score for a candidate lamda and training score for lamda=0 (float, 
                                   fixed at 0.05)
    Attributes:
        lamda_array_ -- a subset array of candidate lamda varying logarithmically between 3 and the candidate lamda 
                        that is next to (and smaller than) the optimal lamda (num_subset_lamda_cand,) 
        valscore_mean_ -- list of average validation scores after totalseed numbers of shuffling for each element of 
                          lamda_array_
        trscore_mean_ -- list of average validation scores after totalseed numbers of shuffling for each element of 
                         lamda_array_
        
    """    
    def __init__(self,datafile='', normalize=True, polydeg='auto', 
                 method='GD', initialize = 'deep', learning_rate='auto', lamda='auto', maxiter=1000,  
                 hidden_units='auto', num_hidden_layers=1, total_units_max=10,  
                 plot_input=False, plot_J=False, report_summary=False, plot_output=False, plot_test=False,                  
                 plot_poly=False, plot_hidden=False, plot_lam=False):
        self.datafile = datafile
        self.normalize = normalize
        self.polydeg = polydeg
        self.method = method 
        self.initialize = initialize
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.maxiter = maxiter
        self.hidden_units = hidden_units        
        self.num_hidden_layers = num_hidden_layers
        self.total_units_max = total_units_max
        self.plot_input = plot_input
        self.plot_J = plot_J
        self.report_summary = report_summary
        self.plot_output = plot_output 
        self.plot_test = plot_test
        self.plot_poly = plot_poly
        self.plot_hidden = plot_hidden
        self.plot_lam = plot_lam
        self.polydeg_max=5
        self.lamda_numdecade=3
        self.totalseed=10 
        self.topscore=0.95 
        self.deltascore=0.05
        self.valscoredelta = 0.005
        self.deltatrscorevslam0 = 0.05
   
    def deepcombine(self, poly=1, lam=0, part=False, hu=[]): 
        """ Returns an instance of class DeepCombine() where the class parameters polydeg, lamda, partition, and 
            hidden_units were employed as function parameters poly, lam, part, hu. This is a supporting function that 
            allows automated selection of optimal polydeg, lamda, and hidden_units parameters by other functions of 
            class DeepLearnAuto()(which would not be possible if class DeepCombine() were to be instantiated under 
            __init__().)
        Parameters:
            poly -- boolean that allows feature matrix to be augmented with polynomial terms if > 1 (int, default = 1)
            lam -- regularization parameter (float >= 0, default = 0)
            part -- boolean that allows partitioning of the dataset into training and validation sets if True 
                    (default = False)
            hu -- list showing number of units in each hidden layer
        Returns:
            dc_instance -- an instance of class DeepCombine() allowing to flexibly set parameters polydeg, lamda, 
                           partition, and hidden_units
        """
        dc_instance = DeepCombine(
                      datafile = self.datafile,
                      normalize = self.normalize,
                      polydeg = poly,
                      partition = part,
                      method = self.method,  
                      initialize  = self.initialize,
                      learning_rate = self.learning_rate,
                      lamda = lam,
                      maxiter = self.maxiter,                
                      hidden_units = hu)
        return dc_instance

    def num_class(self):
        """ Returns number of classes in the dataset.
        Returns:
            number_classes -- number of classes associated with the target vector (int, > 1)
        """
        temp_instance = self.deepcombine() #initialize this object just to get pp.numclass_
        temp_instance.pp.process_training_set();
        number_classes = temp_instance.pp.numclass_
        return number_classes
    
    def candidate_poly(self):
        """ Generates a list of candidate polynomial degrees.
        Returns:
            cand_poly -- list of integers from 1 to polydeg_max
        """
        cand_poly = (np.arange(self.polydeg_max)+1).tolist()
        return cand_poly
    
    def candidate_lamda(self): 
        """ Generates a list of candidate lambda parameters for regularization.
        Returns:
            cand_lamda -- list of floats in descending order varying as .. 0.1, 0.03, 0.01 (= minimum candidate lamda 
                          value) for a number of decades set by parameter lamda_numdecade. Max value = 3 for 
                          lamda_numdecade = 3
        """
        cand_lamda = -np.sort(-np.append(np.logspace(1,self.lamda_numdecade,self.lamda_numdecade)/1000, 
                                         np.logspace(1,self.lamda_numdecade,self.lamda_numdecade)*3/1000)) 
                    # 
        return cand_lamda 
    
    def candidate_hidden_units(self):
        """ Generates a list of candidate hidden unit configurations. The number of hidden layers is set by parameter
            num_hidden_layers. Each hidden layer has the same number of units. The minimum number of units is 
            num_class+1. The maximum number of units is set by parameter total_units_max such that the total number 
            of units in all hidden layers combined cannot exceed total_units_max. As an example, if num_class = 3, 
            num_hidden_layers = 3 and total_units_max = 16, the candidate hidden_layers would be [[4,4,4],[5,5,5]]. 
            In addition, an empty list, which corresponds to logistic regression reference is always added as a 
            candidate.
        Returns:
            cand_hid_lay -- list of candidate hidden layer configurations which themselves are lists
        """
        a = []        
        def b(i):
            return (np.ones((1,self.num_hidden_layers))*(self.num_class()+1+i)).tolist()[0]
            # b(i) returns a list that is num_hidden_layers long and each element is equal to nc+1+i. 
            # Since i >=0, this means the candidate number of units will always be at least 1 higher than num_class
        i = 0
        while 0 < np.sum(b(i)) <= self.total_units_max: # total number of units cannot exceed total_units_max             
            c = list(map(int, b(i)))
            a.append(c) # makes each element of list b(i) an integer
            i += 1
        cand_hid_lay = [[]] + a # always add "no hidden layers" case (logistic regression) as a candidate
        return cand_hid_lay
        
    def scores_vs_poly_or_hu(self, cand_poly_or_hu):
        """ Generates the classification scores for a list of candidate hyperparameters which can be either polynomial 
            degree or hidden units. Auto-detects whether search to be done for polydeg or hidden units based on the 
            input parameter format.
        Parameters:
            cand_poly_or_hu -- a list of integers if candidate hyperparameter is polynomial degree, e.g., [1,2,3,4]
                               a list of lists if candidate hyperparameter is hidden units, e.g., [[],[3,3],[4,4]]
        Returns:
            sum_arr_sorted -- np array showing scores vs candidate hyperparameters in ascending order 
                             (num_final_cand_param,2) 
        """
        scores = []
        param = []
        scr = 0
        i = 0
        while i < len(cand_poly_or_hu) and scr < self.topscore: # stop if topscore is exceeded before going through all 
                                                                # the candidates. topscore can be selected just 1 or as
                                                                # in our case < 1 to potentially end up with a faster 
                                                                # selection and a smaller number of candidates. topscore 
                                                                # < 1 may also have a regularization impact.    
            if type(cand_poly_or_hu[0]) is list:  # this means cand_poly_or_hu is hidden_units
                if self.polydeg == 'auto':
                    pol = 1
                else:
                    pol = self.polydeg
                dc = self.deepcombine(poly=pol,lam=0,part=False, hu = cand_poly_or_hu[i]) # if polydeg='auto', compute 
                                                                                          # scores vs hidden_units for 
                                                                                          # polydeg=1 & lam=0, w/o 
                                                                                          # partitioning the data
                    
            else:  # this means cand_poly_or_hu is polydeg
                if self.hidden_units == 'auto':
                    hU = [] 
                else:
                    hU = self.hidden_units
                dc = self.deepcombine(poly=cand_poly_or_hu[i], lam=0, part=False, hu=hU) # if  hidden_units='auto', 
                                                                                         # compute scores vs polydeg for 
                                                                                         # hidden_units=[] & lam=0 w/o 
                                                                                         # partitioning the data
            solve = dc.combine();                                    
            if solve != None:
                scr = solve['score_tr']
                scores.append(scr)
                param.append(cand_poly_or_hu[i])                
            else:
                para = 'hidden units' if isinstance(cand_poly_or_hu[i], list) == True else 'poly deg'
                print ('no convergence for '+str(para)+" =",cand_poly_or_hu[i])
            i+=1
        if len(param) == 0:
            print ("\nNone of the parameters converged within the search range.\n")
        assert len(param) > 0 # abort if no convergence for any candidate polydeg or hidden unit configuration         
        if param == [[]]: # handle special case 
            summ_arr_sorted = np.array(([],np.array(scores))).reshape((1,2)) # otherwise column_stack does not work
        else:            
            summ_arr = np.column_stack((np.array(param), np.array(scores)))
            summ_arr_sorted = summ_arr[np.argsort(summ_arr[:, 1])] # sort polydeg or hidden_units by their score
        return summ_arr_sorted

    def find_best_poly_or_hu(self, summ_arr_sorted):
        """ Finds the optimal polydeg or hidden units configuration starting with the candidate array sorted by 
            scores (provided by scores_vs_poly_or_hu function.) However the optimal candidate is not simply the one 
            with the highest score. Another candidate with slightly lower score is preferred if it is less complex 
            (lower polydeg or less number of units.) Algorithm here starts with highest ranked candidate and moves  
            to those with lower scores as long as the score difference is lower than parameter deltascore. Once a 
            subset of candidates with scores close enough to the highest score is determined, the candidate with 
            minimum complexity out of that subset is selected as the optimal parameter. 
        Parameters:
            sum_arr_sorted -- np array showing scores vs candidate hyperparameters in ascending order 
                             (num_final_cand_param,2) 
        Returns:
            bestparam -- optimal polydeg (integer) or optimal hidden units configuration (list of integers)        
        """
        s = len(summ_arr_sorted)
        param = summ_arr_sorted[:,0]
        scores = summ_arr_sorted[:,1]
        paramsum = []
        for i in range (0,s):
            paramsum.append(np.sum(param[i])) # paramsum = param for polydeg or hidden units with single hidden unit.                
        if s == 1:
            bestparam = param[0] # if only one candidate available that one is automatically the best
        elif s > 1:
            highest_score = scores[-1] # last score is the highest since the array was sorted
            idx = s-1
            while (highest_score - scores[idx]) < self.deltascore and idx > 0:
                idx -= 1 # If the score difference from one param to next is not bigger than parameter deltascore, this
                         # means complexity can be reduced (polydeg can be decreased or a smaller number of units per 
                         # layer can be used) with a relatively small penalty in acuracy. This is equivalent to 
                         # increasing regularization parameter lamda in that it would tend to prevent overfitting.
            sum_better_scores = paramsum[idx+1:s] # sum is being used as a measure of complexity for hidden layers 
                                                  # with more than one unit  
            param_better_scores = param[idx+1:s] # this is a subset of parameters with better scores. The reason why
                                                 # param[idx+1] does not simply become the best parameter is because
                                                 # within the param_better_scores subset, there may actually be a 
                                                 # a smaller polydeg or hidden layers with less number of units and yet
                                                 # with higher score
            idx_min_sum = np.argmin(sum_better_scores)
            bestparam = param_better_scores[idx_min_sum]
        return bestparam
       
    def plot_scores_vs_poly_or_hu(self, summ_arr_sorted, bestparam): 
        """ Plots scores vs candidate poly degs or vs candidate hidden unit configurations and highlights the optimal
            parameter.
        Parameters:
            sum_arr_sorted -- np array showing scores vs candidate hyperparameters in ascending order 
                              (num_final_cand_param,2) 
            bestparam -- optimal polydeg (integer) or optimal hidden units configuration (list of integers)                         
        """
        param = summ_arr_sorted[:,0].tolist()
        scores = summ_arr_sorted[:,1]
        x = np.arange(0,len(summ_arr_sorted),1).tolist() # x: [0,1,2..len(scores)-1]
        plt.plot(x,scores) # scores vs x
        plt.scatter(x,scores, s=50, c='r', edgecolor='red') # overlay scatter data pts
        index_bestparam = param.index(bestparam)
        score_bestparam = scores[index_bestparam]
        x_bestparam = x[index_bestparam]
        plt.scatter(x_bestparam, score_bestparam, s=100, facecolor="None", edgecolor='black', linewidth=3) # highlight 
                                                                                                           # bestparam        
        plt.xlim(-1,len(scores))
        LABELS = param
        if type(param[0]) is list: 
            xlab = 'hidden units'
            plt.xticks(x, LABELS, rotation=90) # replace x with candidate hidden units
        else:
            xlab = 'polynomial degree' 
            plt.xticks(x, map(int,LABELS)) # replace x with candidate poly degs
        plt.xlabel(xlab)
        plt.ylabel('score')
        plt.title("score vs "+xlab)
        plt.show() 
    
    def mean_scores_of_shuffled_data(self, polydeg, hidden_units, lamda):
        """ Partitions the dataset randomly and differently for the number of times set by parameter totalseed, 
            calculates the training and validation scores for each partitioning and returns the average of them.
            The purpose is to minimize the variation of training and validation scores due to random partitioning. 
        Parameters:
            polydeg -- boolean that allows feature matrix to be augmented with polynomial terms if > 1 (int, default=1)
            hidden_units -- list showing number of units in each hidden layer
            lamda -- regularization parameter (float >= 0, default = 0) 
        Returns:
            mean_val_score -- validation set average score after the dataset is randomly partitioned by totalseed 
                              number of times (float, min:0, max:1)                             
            mean_tr_score -- training set average score after the dataset is randomly partitioned by totalseed 
                             number of times (float, min:0, max:1) 
        """    
        valscores, trscores = [], []
        for seed in range(0,self.totalseed):           
            dc = self.deepcombine(poly=polydeg,lam=lamda,part=True,hu=hidden_units)
            results = dc.combine(seedno=seed);                                    
            if results != None:        
                valscores.append(results['score_val']) 
                trscores.append(results['score_tr'])
            else:
                print ('no convergence for lamda =',lamda, "seed =", seed)
        if len(trscores) == 0:
            print("\nNo convergence for any of the attempted seeds while lamda =", lamda, "\n")
        assert len(trscores) != 0
        mean_val_score, mean_tr_score = np.mean(valscores), np.mean(trscores)
        return mean_val_score, mean_tr_score
        
    def find_bestlambda(self, bestpoly, besthu):
        """ Selects the optimal lambda. Starts by collecting mean training and validation scores for lamda = 10. Then
            reduces lamda logarithmically as long as validation score improves by an amount higher than the parameter 
            valscoredelta OR training score is too small (i.e., difference between training score for the actual lamda 
            and training score for lamda=0 is higher than the parameter deltatrscorevslam0.) Optimal lamda is the last 
            one before both these conditions become invalid. If the conditions become valid down to the smallest 
            lamda, which is 0.01, then the optimal lamda is 0.01.            
        Parameters:
            bestpoly -- optimal polydeg (integer)
            besthu -- optimal hidden units configuration (list of integers)    
        Returns:
            bestlambda -- optimal lamda (float, >= 0.01)
        """
        ref_scores = self.mean_scores_of_shuffled_data(bestpoly,besthu,lamda=0) 
        trscore0_mean=ref_scores[1] # tr score for lamda=0
        lamda_cand = self.candidate_lamda()
        laminit = 10 # next one higher than the nominal max, which is 3 for lamda_numdecade = 3
        init_scores = self.mean_scores_of_shuffled_data(bestpoly,besthu,lamda=laminit)
        self.valscore_mean_, self.trscore_mean_=[init_scores[0]],[init_scores[1]]  
        lamd=[laminit]
        i = 0
        while i < len(lamda_cand): # the highest candidate is 3, lowest 0.01
            scores = self.mean_scores_of_shuffled_data(bestpoly,besthu,lamda_cand[i])            
            self.valscore_mean_.append(scores[0])
            self.trscore_mean_.append(scores[1]) 
            lamd.append(lamda_cand[i])            
            if (self.valscore_mean_[i+1]-self.valscore_mean_[i] > self.valscoredelta or 
                trscore0_mean - self.trscore_mean_[i+1] > self.deltatrscorevslam0):
            # reduce lamda as long as valscore improvement is above certain threshold (%0.5, just to have it slightly 
            # above 0.) The 2nd condition (following or) prevents a case where there is no validation score improvement 
            # for decreasing lamda while lamda is very large (e.g., valscore being flat from lam = 10 to 3)
                i += 1
            else:
                break
        self.lamda_array_ = np.array(lamd)
        bestlamda = self.lamda_array_[-2] 
        return bestlamda
         
    def plot_scores_vs_lamda(self):
        """ Plots the average training and validation scores as a function of a subset of candidate lamda. 
            Training and validation scores corresponding to optimal lamda were also highlighted.
        """
        lamda = self.lamda_array_
        valscores = self.valscore_mean_
        trscores = self.trscore_mean_
        bestvalscore = self.valscore_mean_[-2]
        besttrscore = self.trscore_mean_[-2]
        bestlambda = self.lamda_array_[-2]
        plt.semilogx(lamda,valscores,label='val')
        plt.semilogx(lamda,trscores,label='tr')            
        plt.scatter(lamda,valscores, s=50, c='r', edgecolor = 'r')
        plt.scatter(lamda,trscores, s=50, c='blue', edgecolor='blue')
        plt.scatter(bestlambda, bestvalscore, s=100, facecolor="None", edgecolor='black', linewidth=3)        
        plt.scatter(bestlambda, besttrscore, s=100, facecolor="None", edgecolor='black', linewidth=3)
        plt.xscale('log')
        plt.legend(loc='best')
        plt.xlabel("lambda")
        plt.ylabel("scores")
        plt.title("scores vs lambda")
        plt.show() 
    
    def hyperparams(self):
        """ Combines functions to compute optimal polydeg, hidden units configuration, and lamda in auto mode and
            displays optionally the associated plots. If all three parameters are in the auto mode, optimal polydeg 
            is calculated first, then the optimal hidden units configuration is calculated by using the optimal polydeg
            and finally the optimal lamda is calculated using the optimal polydeg and optimal hidden units. If any of 
            these hyperparameters is not in auto mode, the value defined by the user is employed. 
        Returns:
            bestpoly -- optimal polydeg (integer)
            besthu -- optimal hidden units configuration (list of integers) 
            bestlamda -- optimal lamda (float, >= 0.01)
        """
        if self.polydeg == 'auto':
            cand = self.candidate_poly()
            arr_poly = self.scores_vs_poly_or_hu(cand)            
            bestp = self.find_best_poly_or_hu(arr_poly)
            bestpoly = np.int(bestp)
            self.polydeg = bestpoly # so that arr_hu = self.scores_vs_poly_or_hu(cand) below uses bestpoly value
            if self.plot_poly == True:
                self.plot_scores_vs_poly_or_hu(arr_poly, bestpoly) 
        else:
            bestpoly = self.polydeg         
        if self.hidden_units == 'auto':
            cand = self.candidate_hidden_units()
            arr_hu = self.scores_vs_poly_or_hu(cand)
            besthu = self.find_best_poly_or_hu(arr_hu)
            if self.plot_hidden == True:
                self.plot_scores_vs_poly_or_hu(arr_hu, besthu) 
        else:           
            besthu = self.hidden_units        
        if self.lamda == 'auto':
            bestlamda = self.find_bestlambda(bestpoly,besthu)
            if self.plot_lam == True:
                self.plot_scores_vs_lamda()
        else:
            bestlamda = self.lamda
        return bestpoly, besthu, bestlamda
    
    def fit_auto(self):
        """ First calculates optimal hyperparameters that are in auto mode. Next, using those hyperparameters, finds
            theta with an additional round of iterations while plotting optionally related graphs and creating the 
            summary report. In addition to returning the number of iteraions, execution time, and the final cost, if 
            the training set is normalized, makes available its mean and sigma values, which can be used for prediction
            on the test set.
        Returns:
            theta -- dictionary holding network parameters
                     keys: int > 0, total number of keys: num_layers - 1
                     dimensions for each key: (num_units_next, num_units_prev+1)
            score_tr -- number of accurately predicted labels over the number of all predictions on the training set
                        (float, min:0, max:1) 
            bestpoly -- optimal polydeg (integer)
            besthu -- optimal hidden units configuration (list of integers) 
            bestlamda -- optimal lamda (float, >= 0.01)
            niter -- number of iterations (int > 0)
            timetofit -- execution time taken by solver (sec)
            finalcost -- cost after the last iteration
            mu_tr -- array containing normalization mean from the training set, an empty list in case taining set is 
                     not normalized (num_features,)  
            sigma_tr -- array containing normalization standard deviation from the training set, an empty list in case 
                        training set is not normalized (num_features,)
        """
        bestpoly, besthu, bestlambda = self.hyperparams();
        best_inst = self.deepcombine(poly = bestpoly,lam = bestlambda, part=False, hu = besthu) 
        param = best_inst.combine(plot_input=self.plot_input, report_summary=self.report_summary, 
                                  plot_output=self.plot_output, plot_J=self.plot_J);                            
        theta, score_tr, niter, timetofit, finalcost = (param['theta'], param['score_tr'], param['numiter'], 
                                                       param['timetofit'], param['finalcost'])
        if self.normalize==True:
            mu_tr = best_inst.pp.mu_
            sigma_tr = best_inst.pp.sigma_
        else:
            mu_tr = []
            sigma_tr = []
        return theta, score_tr, bestpoly, besthu, bestlambda, niter, timetofit, finalcost, mu_tr, sigma_tr
    
    def pred_score_test(self, test_file, theta, bestpoly, besthu, mu_tr,sigma_tr):
        """ Pre-processes the test set based on whether the training set was normalized or had used a polydeg > 1. 
            Then, using theta and hidden units configuration, makes prediction on the test set and finally provides
            the score associated with the test set.
        Parameters:
            test_file -- test data set never seen by the algorithm while theta is found 
                        (num_test_examples, original number of features in the training set)           
            theta -- dictionary holding network parameters
                     keys: int > 0, total number of keys: num_layers - 1
                     dimensions for each key: (num_units_next, num_units_prev+1)
            bestpoly -- optimal polydeg (integer)
            besthu -- optimal hidden units configuration (list of integers) 
            mu_tr -- array containing normalization mean from tr set, an empty list in case tr set is not normalized
                    (num_features,)  
            sigma_tr -- array containing normalization std dev from tr set, an empty list in case tr set is not 
                        normalized (num_features,)
        Returns:
            pred_test -- array of predicted output on the test set using theta that was determined by a separate 
                         training set. This is in standard form of int >= 0 (num_test_samples,)
            score_test -- number of accurately predicted labels over the number of all predictions on the test set
                          (float, min:0, max:1)                        
        """
        best_inst = self.deepcombine(bestpoly,hu=besthu)
        self.normalize == True if len(mu_tr)!=[] and len(sigma_tr)!=[] else False
        Xtest, ytest = best_inst.pp.process_test_set(test_file,mu_tr,sigma_tr)
        pred_test = best_inst.ds.predict(theta,Xtest);
        if self.plot_test == True:
            best_inst.plt_output(Xtest,ytest,theta,pred_test,tr=False)
            plt.close()
        score_test = best_inst.ds.score(pred_test,ytest)
        score_test = np.round(score_test,3)
        return pred_test,score_test

    def generate_summary(self):
        """ Combines fit_auto and pred_score_test methods to return a dataframe which displays essentially all
            parameters of interest: datafile short name, normalization, calculation method (algorithm), number of 
            iterations, execution time, optimal polynomial degree, optimal hidden units configuration, optimal lamda, 
            training score, and test score (so an independent test file should be available.) If training set filename 
            is abc_tr.csv, test set should be named abc_test.csv and should be inserted under the same folder as the 
            training set.
        Returns:
            scoreboard_df -- summary table (dataframe)                      
        """
        theta,score_tr, bestpoly, besthu, bestlamda, niter, timetofit, finalcost, mu_tr, sigma_tr = self.fit_auto()
        ### datafile string manipulation to get the concise and correct filenames
        slash_loc1 = np.char.find(self.datafile,'\\')
        temp = self.datafile[slash_loc1+1:]
        slash_loc2 = np.char.find(temp,'\\')
        _loc = np.char.find(temp,'_tr')        
        self.dataname = temp[slash_loc2+1:_loc]
        testfile = self.datafile[:_loc+slash_loc1+1]+"_test.csv"
        ###
        ps_test = self.pred_score_test(testfile, theta, bestpoly, besthu, mu_tr, sigma_tr)
        scoreboard = np.array((self.dataname,self.normalize, self.method, niter, timetofit, bestpoly,str(besthu), 
                               bestlamda, np.round(score_tr,3),ps_test[1].astype(str))).reshape((1,10))
        cols = ['data','normalize','method','num iter','exec time','poly deg','hidden layers','lamda', 'tr score', 
                'test score']
        scoreboard_df = pd.DataFrame(scoreboard,columns = cols)
        return scoreboard_df

