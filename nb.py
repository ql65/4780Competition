import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from scipy.io import loadmat
import csv


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

y = []
for i in range(0, 10):
    y.append(yTr[:, i])

yLabel = np.zeros((yTr.shape[0],))
index = []
for i in range(0, 10):
    index.append(y[i] == 1)
    yLabel[index[i]] = i
print yLabel

def naivebayesPY(x,y):
    """
        function [pos,neg] = naivebayesPY(x,y);
        
        Computation of P(Y)
        Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)
        
        Output:
        pos: probability p(y=1)
        neg: probability p(y=-1)
        """
    
    # add one example of each class to avoid division by zero ("plus-one smoothing")
    y = np.concatenate([y, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    n = len(y)
    prob = np.zeros((10, ))
    for i in range(0, 10):
        pos = y[y == i]
        prob[i] = (float)(len(pos))/(float)(n)
    return prob

def naivebayesPXY(x,y):
    """
        Computation of P(X|Y)
        Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)
        
        Output:
        posprob: probability vector of p(x|y=1) (dx1)
        negprob: probability vector of p(x|y=-1) (dx1)
        """
    
    # add one per digit example to avoid division by zero ("plus-one smoothing")
    n, d = x.shape
    n, d = x.shape
    XiY = np.zeros((10, d))
    #TODO: find uac and sig(ac) to calculate P(xa | y = c)
    for i in range(0, 10):
        y_i = (y == i)
        x_i = x[y_i]
        total_i = np.sum(np.count_nonzero(x_i))
        x_i_sum = (x_i != 0).sum(0)
        XiY[i] = np.divide(x_i_sum, (float)(total_i))
    return XiY

def naivebayes(x,y,xtest):
    """
        function logratio = naivebayes(x,y);
        
        Computation of log P(Y|X=x1) using Bayes Rule
        Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1)
        xtest: input vector of d dimensions (1xd)
        
        Output:
        logratio: log (P(Y_i = 1|X=x1)
        """
    print xtest
    (n, d) = x.shape
    PXiY = naivebayesPXY(x, y)
    PY = naivebayesPY(x, y)
    
    x_i_one = (x == 1).sum(0)
    x_i_zero = (x == 0).sum(0)
    x_i_one = np.divide(x_i_one, (float)(n))
    x_i_zero = np.divide(x_i_zero, (float)(n))

    XtestOne = (xtest == 1)
    XtestZero = (xtest == 0)
    Xone = x_i_one[XtestOne]
    Xzero = x_i_zero[XtestZero]
    PXone = np.prod(Xone)
    print "pxone", PXone
    PXzero = np.prod(Xzero)
    px = PXone * PXzero
    print px
    pyxi = np.zeros((10, ))
    for i in range(0, 10):
        print np.power(PXiY[i], xtest)
#pyxi[i] = (np.prod(np.power(PXiY[i], xtest)) * PY[i]) / (float)(px)
#XtestOne = (xtest == 1)
#XtestZero = (xtest == 0)
#Xone = XOne[XtestOne]
#Xzero = XZero[XtestZero]
#PXone = np.prod(Xone)
#PXzero = np.prod(Xzero)
#PX = PXone * PXzero
    
#PYXP = (np.prod(np.power(PXYP, xtest)) * PYP) / (float)(PX)
#PYXN = (np.prod(np.power(PXYN, xtest)) * PYN) / (float)(PX)
#return np.log(PYXP/(float)(PYXN))
    print pyxi
    return pyxi


naivebayes(xTr, yLabel, xTe[0])
