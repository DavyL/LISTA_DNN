import numpy as np


##ISTA_solver : Computes all iterations of ISTA
# A : the dictionary matrix
# y : observed signal
# k_max : the maximal number of iterations
# err : another condition to break the loop (Not impl yet)
# Returns x_hat, the estimated solution
##Notes : if L is set to the largest eigenvalue of the gram matrix A^T*A, then things go well,
#if this is not the case and lambda is too small, then nonsense happens, the residual  goes to infinity and everything breaks


def ISTA_solver(A, obs, k_max = 10, err = 0.1, lambd = 1, L = 1):
    x_hat = np.zeros(len(A[0]))     ##Approximated solution
    res = obs                         ##Residual y - Ax
    res_val = np.zeros(k_max)
    soft_thresh = lambda x : np.sign(x)*max(0, abs(x) - lambd/L)       
    for k in range(k_max):
        x_hat = ISTA_iter(A, obs, x_hat, soft_thresh, L)
        res = obs - np.matmul(A,x_hat) 
        res_val[k] = np.dot(res, res)
    return x_hat, res_val

##ISTA_iter : performs one step of ISTA
def ISTA_iter(A, obs, x_hat, thresh_func, L):
    #print("residual in ISTA_iter :" + str(res))
    #print("length of residual : " + str(len(res)))
    #print("x_hat in ISTA_iter :" + str(x_hat))
    #print("length of x_hat : " + str(len(x_hat)))
    unthresh_x = x_hat + (1/L)*np.matmul(A.transpose(), obs - np.matmul(A, x_hat))
    thresh_x = np.zeros(len(unthresh_x))
    for i in range(len(thresh_x)):
        thresh_x[i] = thresh_func(unthresh_x[i])
    return thresh_x


##FISTA_solver : initializes and computes iterates of FISTA
def FISTA_solver(A, obs, k_max = 10, err = 0.1, lambd = 1, L = 1):
    res = np.zeros(k_max)
    y = np.zeros(len(A[0]))
    x = np.zeros(len(A[0]))     
    t = 1
    soft_thresh = lambda x : np.sign(x)*max(0, abs(x) - lambd/L)       
    for k in range(k_max):
        x, t, y = FISTA_iter(A, obs, x, t, y, soft_thresh, L)
        res[k] = np.dot(obs - np.matmul(A,x), obs - np.matmul(A,x)) 

    return x, res

##FISTA_iter : performs one step of FISTA
#FISTA differs from ISTA as it updates the residual at each step depending on some parameters t and y 
#the new x is computed from y 
#the new y is computed from a linear combination involving t and x (both old and new)
#Returns updated x, t and y
def FISTA_iter(A, obs, x_hat, t, y, thresh_func, L):
    #print("residual in ISTA_iter :" + str(res))
    #print("length of residual : " + str(len(res)))
    #print("x_hat in ISTA_iter :" + str(x_hat))
    #print("length of x_hat : " + str(len(x_hat)))
    new_x = ISTA_iter(A, obs, y, thresh_func, L)
    

    new_t = (1+np.sqrt(1+4*(t**2)))/2
    new_y = new_x + ((t-1)/new_t)*(new_x - x_hat)
    return new_x, new_t, new_y

