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
    print prob
    return prob

def naivebayesPXY(x,y):
    """
        function [posprob,negprob] = naivebayesPXY(x,y);
        
        Computation of P(X|Y)
        Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)
        
        Output:
        posprob: probability vector of p(x|y=1) (dx1)
        negprob: probability vector of p(x|y=-1) (dx1)
        """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    n, d = x.shape
    x = np.concatenate([x, np.ones((2,d))])
    y = np.concatenate([y, [-1,1]])
    n, d = x.shape
    yPos = (y == 1)
    yNeg = (y == -1)
    Xpos = x[yPos]
    Xneg = x[yNeg]
    totalPos = np.sum(np.count_nonzero(Xpos))
    totalNeg = np.sum(np.count_nonzero(Xneg))
    XiPos = (Xpos != 0).sum(0)
    XiNeg = (Xneg != 0).sum(0)
    XiPos = np.divide(XiPos, (float)(totalPos))
    XiNeg = np.divide(XiNeg, (float)(totalNeg))
    return (XiPos, XiNeg)

def naivebayes(x,y,xtest):
    """
        function logratio = naivebayes(x,y);
        
        Computation of log P(Y|X=x1) using Bayes Rule
        Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1)
        xtest: input vector of d dimensions (1xd)
        
        Output:
        logratio: log (P(Y = 1|X=x1)/P(Y=-1|X=x1))
        """
    (n, d) = x.shape
    (PXYP, PXYN) = naivebayesPXY(x, y)
    (PYP, PYN) = naivebayesPY(x, y)
    XZero = (x == 0).sum(0)
    XOne = (x == 1).sum(0)
    XOne = np.divide(XOne, (float)(n))
    XZero = np.divide(XZero, (float)(n))
    XtestOne = (xtest == 1)
    XtestZero = (xtest == 0)
    Xone = XOne[XtestOne]
    Xzero = XZero[XtestZero]
    PXone = np.prod(Xone)
    PXzero = np.prod(Xzero)
    PX = PXone * PXzero
    
    PYXP = (np.prod(np.power(PXYP, xtest)) * PYP) / (float)(PX)
    PYXN = (np.prod(np.power(PXYN, xtest)) * PYN) / (float)(PX)
    return np.log(PYXP/(float)(PYXN))

def naivebayesCL(x,y):
    """
        function [w,b]=naivebayesCL(x,y);
        Implementation of a Naive Bayes classifier
        Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1)
        
        Output:
        w : weight vector of d dimensions
        b : bias (scalar)
        """
    
    n, d = x.shape
    [xpos, xneg] = naivebayesPXY(x,y)
    [ypos, yneg] = naivebayesPY(x, y)
    w = np.log(xpos) - np.log(xneg)
    b = np.log(ypos) - np.log(yneg)
    return (w, b)

#</GRADED>

naivebayesPY(xTr,yLabel)

