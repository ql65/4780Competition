
# coding: utf-8

# In[50]:

import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from cvxpy import *
import csv
from l2distance import l2distance


# In[51]:

#Load Raw Data
trainfile = loadmat('train.mat', squeeze_me=True, struct_as_record=False)
testfile = loadmat('test.mat', squeeze_me=True, struct_as_record=False)
xTr = trainfile['x']
#xValid = xTr[2000: -1]
xTr = xTr[0 : 1999]
yTr = trainfile['y']
index = yTr == 0
yTr[index] = -1
print xTr.shape
print yTr.shape
#yValid = yTr[2000: -1]
yTr = yTr[0: 1999]
xTe = testfile['x']


# In[52]:

#Train
def primalSVM(xTr, yTr, C=1):
    """
    function (classifier,w,b) = primalSVM(xTr,yTr;C=1)
    constructs the SVM primal formulation and uses a built-in 
    convex solver to find the optimal solution. 
    
    Input:
        xTr   | training data (nxd)
        yTr   | training labels (nx1)
        C     | the SVM regularization parameter
    
    Output:
        fun   | usage: predictions=fun(xTe);
        wout  | the weight vector calculated by the solver
        bout  | the bias term calculated by the solver
    """
    N, d = xTr.shape
    y = yTr.flatten()
    print xTr.shape
    print yTr.shape
    w = Variable(d)
    b = Variable(1)
    ones = np.full((N, 1), 1) 
    slack = pos(ones - (y * diag(xTr * w + b * ones)).T)
    objective = sum_squares(w) + C * sum_entries(slack) 
    prob = Problem(Minimize(objective))
    prob.solve()
    wout = w.value
    bout = b.value
    
    fun = lambda x: x.dot(wout) + bout
    return fun

def computeK(kerneltype, X, Z, kpar=0):
    """
        function K = computeK(kernel_type, X, Z)
        computes a matrix K such that Kij=g(x,z);
        for three different function linear, rbf or polynomial.
        
        Input:
        kerneltype: either 'linear','polynomial','rbf'
        X: n input vectors of dimension d (nxd);
        Z: m input vectors of dimension d (mxd);
        kpar: kernel parameter (inverse kernel width gamma in case of RBF, degree in case of polynomial)
        
        OUTPUT:
        K : nxm kernel matrix
        """
    assert kerneltype in ["linear","polynomial","poly","rbf"], "Kernel type %s not known." % kerneltype
    assert X.shape[1] == Z.shape[1], "Input dimensions do not match"
    
    # TODO 2
    if kerneltype == "linear":
        K = np.dot(X, Z.T)
    elif kerneltype == "polynomial" or kerneltype == "poly":
        product = np.dot(X, Z.T)
        ones = np.full(product.shape, 1)
        K = np.power(np.add(ones, np.dot(X, Z.T)), kpar)
    elif kerneltype == "rbf":
        pair_distance = l2distance(X, Z)
        K = np.exp(-pair_distance * pair_distance * kpar)
    return K
#</GRADED>

#<GRADED>
def dualqp(K,yTr,C):
    """
        function alpha = dualqp(K,yTr,C)
        constructs the SVM dual formulation and uses a built-in
        convex solver to find the optimal solution.
        
        Input:
        K     | the (nxn) kernel matrix
        yTr   | training labels (nx1)
        C     | the SVM regularization parameter
        
        Output:
        alpha | the calculated solution vector (nx1)
        """
    print "dualqp"
    print C
    print yTr
    
    y = yTr.flatten()
    N, _ = K.shape
    alpha = Variable(N)
    
    # TODO 3:
    quad = quad_form((diag(alpha) * y), K)
    objective = 0.5 * sum_entries(quad) - sum_entries(alpha)
    constraints = [alpha >= 0, alpha <= C, alpha.T * y == 0]
    print "before solve"
    prob = Problem(Minimize(objective), constraints)
    prob.solve()
    return np.array(alpha.value).flatten()
#</GRADED>

#<GRADED>
def recoverBias(K,yTr,alpha,C):
    """
        function bias=recoverBias(K,yTr,alpha,C);
        Solves for the hyperplane bias term, which is uniquely specified by the
        support vectors with alpha values 0<alpha<C
        
        INPUT:
        K : nxn kernel matrix
        yTr : 1xn input labels
        alpha  : nx1 vector of alpha values
        C : regularization constant
        
        Output:
        bias : the scalar hyperplane bias of the kernel SVM specified by alphas
        """
    
    # TODO 4
    print "recoverBias"
    y = yTr.reshape((yTr.shape[0], 1))
    a = alpha.reshape((alpha.shape[0], 1))
    N,d = y.shape
    c_half = (float)(C) / 2
    
    best_alpha_index = np.argmin(np.absolute(c_half - a))
    best_alpha = np.amin(c_half - a)
    best_y = y[best_alpha_index]
    best_k = K[best_alpha_index]
    best_y = best_y.reshape((best_y.shape[0], 1))
    best_k = best_k.reshape((best_k.shape[0], 1))
    yA = np.multiply(a, y)
    b = best_y - np.dot(yA.T, best_k)
    return b

#<GRADED>
def dualSVM(xTr,yTr,C,ktype,lmbda):
    """
        function classifier = dualSVM(xTr,yTr,C,ktype,lmbda);
        Constructs the SVM dual formulation and uses a built-in
        convex solver to find the optimal solution.
        
        Input:
        xTr   | training data (nxd)
        yTr   | training labels (nx1)
        C     | the SVM regularization parameter
        ktype | the type of kernelization: 'rbf','polynomial','linear'
        lmbda | the kernel parameter - degree for poly, inverse width for rbf
        
        Output:
        svmclassify | usage: predictions=svmclassify(xTe);
        """
    print "dualSVM"
    y = yTr.reshape((yTr.shape[0], 1))
    k = computeK(ktype, xTr, xTr, lmbda)
    alpha = dualqp(k, yTr, C)
    alpha = alpha.reshape((alpha.shape[0], 1))
    b = recoverBias(k,yTr,alpha,C)
    svmclassify = lambda x: (np.dot(computeK(ktype, xTr, x, lmbda).T, np.multiply(alpha, y)) + b).reshape(((np.dot(computeK(ktype, xTr, x, lmbda).T, np.multiply(alpha, y)) + b).shape[0],))
    return svmclassify
#</GRADED>


y = []
for i in range(0, 10):
    y.append(yTr[:, i])
print len(y), len(y[0])

classifier= []
for i in range(0, 10):
    print i
#classifier.append(dualSVM(xTr, y[i], 10, "rbf", 0.25))
    classifier.append(primalSVM(xTr, y[i], 20))

#Test
yTe = []
for i in range(0, 9):
    yTe.append(np.sign(classifier[i](xTe)))
file = open('yTe.csv', 'w')
writer = csv.writer(file)
for val in yTe:
    writer.writerow(val)
yTe = np.asarray(yTe)
yTe = yTe.T
print yTe[0]