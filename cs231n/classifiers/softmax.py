from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train): 
        #Calculate class scores
        scores = X[i].dot(W)        
        correct_class_score = scores[y[i]]
        
        #Final exp values for each class (needed to calculate local gradient)
        exps = np.exp(scores)
        #denominator is used multiple times, so stored in this var
        denominator = np.sum(exps)
        #Syi is all that is needed for forward pass
        Syi = exps[y[i]]/denominator
        #Update loss with Syi
        loss -= np.log( Syi )
        
        #Local gradient of softmax node is a vector of length numClasses
        dSoftMax = np.zeros( (1, num_classes) )
        #Loop is used to populate vector based on derivative of softmax function
        for j in range(num_classes):
            dSoftMax[0,j] = -Syi * exps[j] / denominator
        #Based on derivative of softmax function, we must perform this operation for the right class
        dSoftMax[0,y[i]] += Syi;
        #This calulation is based on upstream gradient due to -ve log operation. 
        dSoftMax /= -Syi
        
        #Solution key: make computational graph with softmax as a single node and use derivation for derivative of softmax function
        dW += np.reshape(X[i], (X.shape[1],1)).dot(dSoftMax)            
                
    loss /= num_train
    dW /= num_train
    
    loss += reg * np.sum(W * W)
    dW += (reg * 2 * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
