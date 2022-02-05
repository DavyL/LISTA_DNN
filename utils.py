import numpy as np
import matplotlib.pyplot as plt

##Generate_gaussian_dictionary(): Creates a dictionary with gaussian entries
# n_cols, n_rows: number of columns (atoms) and rows (signal dimension)
# se : standard error
#
#Returns : A, the dictionary

def Generate_gaussian_dictionary(n_cols, n_rows, se):
    A = np.random.normal(loc = 0, scale = se, size = (n_rows, n_cols))
    A = A.transpose()
    for c in range(len(A)):
        A[c] *= 1/np.sqrt(np.dot(A[c], A[c]))
    A = A.transpose()
    print("n_rows : " + str(n_rows) + " len A[0]" + str(len(A[0])))
    return A


##Create_sparse_x() : Creates a sparse vector (with 0 and 1s)
# x : length of signal
# number_of_nz : number of nonzero coefficients
#Returns the sparse vector
#Warning : less than number_of_nz nonzero coefficients might be set to 1
def Create_sparse_x(length, number_of_nz):
    x = np.zeros(length)
    for i in range(1,number_of_nz):
        new_index = np.random.randint(0,length)
        x[new_index] = 1
    return x

##Step function() turns every value of x such that |x[i]|> threshold to x[i] = sign(x[i]) * 1
def Step_function(x, threshold):
    for i in range(len(x)):
        if abs(x[i]) > threshold:
            x[i] = np.sign(x[i])
        else :
            x[i] = 0
    return x

##Cross validation computes the best hyperparameter for the square loss function
#Y : observed signal
#A : possible atoms for the signal
#solver : the function(y, hyp, A) that obtains a model
#hyperparameters : a list of hyperparameters
#nfolds : the number of folds to use to cross validate

##Behaviour is as follows : 
#Signal is split in nfolds blocks, each of size len(size)/nfolds (excess signal is dropped if len(size) and nfolds are relatively primes)
#Then nfolds iterates are computed, where one at a time each block becomes a test blocks and other blocks are train blocks
#An estimated solution is computed for given hyperparameter on the train blocks
#The nfolds residuals are added together to give the cross validation value at a given hyperparameter
#The hyperparameter returned is the one minimising sum of residuals

##Note: if solver requires more arguments than only the hyperparameter, solver can be turned into a hyperparameter only depending function by casting solver in a lambda:
#Explicitly, if orig_solver = f(parameter, hyperparameter), then set solver = lambda hyperparameter : f(fixed_param, hyperparameter)
#Since parameter should be fixed across CV this should be ok

def Cross_validation(Y, A, solver, hyperparameters, nfolds, plot = False):
    if(len(Y)%nfolds != 0):
        print("In Cross_validation(): length of observed signal and number of folds are relatively primes, some observations will be dropped")

    split_len = int(len(Y)/nfolds)
    RSS = np.zeros(len(hyperparameters))
    for j in range(len(hyperparameters)):
        hyp = hyperparameters[j]
        print("Progress of cross-validation :" + str(j) + "/" + str(len(hyperparameters)))
        for i in range(nfolds):
            test_indices = list(range(i*split_len, (i+1)*split_len))
            train_indices = list(range(0,i*split_len))
            train_indices.extend(list(range((i+1)*split_len, len(Y))))
            test_Y = Y[test_indices]
            test_A = A[test_indices]
            train_Y = Y[train_indices]
            train_A = A[train_indices]

            x_hat = solver(train_Y, hyp, train_A)[0]
            res = test_Y - np.matmul(test_A,x_hat)
            RSS[j] += np.dot(res, res)
    if plot:
        plt.plot(RSS)
        plt.show()

    best_hyp_index = Get_min_index(RSS)
    return hyperparameters[best_hyp_index]

def Get_min_index(l):
    min_val = min(l)
    for i in range(len(l)):
        if l[i] <= min_val:
            return i
    print("In Get_min_index(): Something went wrong (min index not reached), returning 0")
    return 0
