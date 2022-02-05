import utils as utils
from ISTA import ISTA_solver, FISTA_solver
import numpy as np
import matplotlib.pyplot as plt

##Any function is this file should be working without any additional parameters
#Example : Test_example_FISTA() is a valid command


##Test_example_FISTA: computes FISTA with various parameters
#All parameters are optional
#show_plots: boolean value on whether or not to display various plots
#cv: boolean value on whether or not to perform cross-validation to finetune lambda
#lambd: value of lambda (overriden by cross validated lambda if cv == True)
#dict, x_0: Fixed problem
def Test_example_FISTA(show_plots = False, cv = False, lambd = 0.1, dict = None, obs = None, sparsity = 10, n_cols = 200, n_rows = 100, k_max = 100):
    if dict is None:
        dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)
    if obs is None:
        x_0 = utils.Create_sparse_x(len(dict[0]), sparsity)
        obs = np.matmul(dict, x_0)

    max_eig = max(np.linalg.eigvals(np.matmul(dict.transpose(), dict)))

    if cv:
        FISTA_solver_fixed_param = lambda y, lambd, A : FISTA_solver(A = A, obs = y, k_max = k_max, L=max_eig, lambd = lambd)
        hyperparams_list = np.linspace(0,5,200)
        cv_lambda = utils.Cross_validation(Y = obs, A = dict, solver = FISTA_solver_fixed_param, nfolds=2, hyperparameters=hyperparams_list, plot=True)
        print("In Test_example_FISTA(): lambda selected by cross-validation : " + str(cv_lambda))
        lambd = cv_lambda

    (x_hat, res) = FISTA_solver( A = dict, obs = obs, k_max=k_max, L = max_eig, lambd = lambd)
    x_step = x_hat.copy()
    x_step = utils.Step_function(x_step, 0.1)


    if show_plots:
        plt.figure()
        plt.title('original atoms (g), recovered atoms (b), sign support of recovered atoms (r)')
        plt.plot(x_0, 'g-')
        plt.plot(x_hat, 'b+')
        plt.plot(x_step, 'r+')
        plt.show()

        plt.figure()
        plt.title('original (observed) signal (g) and recovered signal (b)')
        plt.plot(np.matmul(dict,x_0), 'g-')
        plt.plot(np.matmul(dict,x_hat), 'b+')
        plt.show()

        plt.figure()
        plt.title('Convergence of FISTA')
        plt.plot(res, 'g-')
        plt.show()

    return res, x_hat, x_step

##Test_example_ISTA: computes ISTA with various parameters
#All parameters are optional
#show_plots: boolean value on whether or not to display various plots
#cv: boolean value on whether or not to perform cross-validation to finetune lambda
#lambd: value of lambda (overriden by cross validated lambda if cv == True)
#dict, obs: Fixed problem
def Test_example_ISTA(cv = False, lambd = 0.1, show_plots = False, dict = None, obs = None, sparsity = 10, n_cols = 200, n_rows = 100, k_max = 100):
    if dict is None:
        dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)
    if obs is None:
        x_0 = utils.Create_sparse_x(len(dict[0]), sparsity)
        obs = np.matmul(dict, x_0)

    max_eig = max(np.linalg.eigvals(np.matmul(dict.transpose(), dict)))
    if cv:
        ISTA_solver_fixed_param = lambda obs, lambd, A : ISTA_solver(A = A, obs = obs, k_max = k_max, L=max_eig, lambd = lambd)
        hyperparams_list = np.linspace(0,5,200)
        cv_lambda = utils.Cross_validation(Y = obs, A = dict, solver = ISTA_solver_fixed_param, nfolds=2, hyperparameters=hyperparams_list, plot=True)
        print("In Test_example_ISTA() : lambda selected by cross-validation : " + str(cv_lambda))
        lambd = cv_lambda


    (x_hat, res) = ISTA_solver(A = dict, obs = obs, k_max=k_max, L = max_eig, lambd = lambd)
    x_step = x_hat.copy()
    x_step = utils.Step_function(x_step, 0.1)


    if show_plots:
        plt.figure()
        plt.title('original atoms (g), recovered atoms (b), sign support of recovered atoms (r)')
        plt.plot(x_0, 'g-')
        plt.plot(x_hat, 'b+')
        plt.plot(x_step, 'r+')
        plt.show()

        plt.figure()
        plt.title('original (observed) signal (g) and recovered signal (b)')
        plt.plot(np.matmul(dict,x_0), 'g-')
        plt.plot(np.matmul(dict,x_hat), 'b+')
        plt.show()
        
        plt.figure()
        plt.title('Convergence of ISTA')
        plt.plot(res, 'g-')
        plt.show()

    return res, x_hat, x_step


##Compare_speeds: Computes the step by step residuals of ISTA and FISTA
#The problem is fixed for both methods so this should be used instead of independent calls to ..._ISTA and ..._FISTA
def Compare_speeds(n_cols = 200, n_rows = 100, sparsity = 10, cv = False, max_iter = 100):
    dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)
    x_0 = utils.Create_sparse_x(len(dict[0]), sparsity)
    obs = np.matmul(dict, x_0)

    (res_ista, x_ista, x_ista_step) = Test_example_ISTA(cv = cv, show_plots = False, dict = dict, obs = obs, k_max = max_iter)
    (res_fista, x_fista, x_fista_step) = Test_example_FISTA(cv = cv, show_plots = False, dict = dict, obs = obs, k_max = max_iter)

    plt.figure()
    plt.title('Comparison of ISTA (green) and FISTA (blue)')
    plt.subplot(221)
    plt.title('Convergence of ISTA (g) and FISTA (b)')
    plt.plot(res_ista, 'g-')
    plt.plot(res_fista, 'b-')

    plt.subplot(222)
    plt.title('True atoms (r) and recovered atoms of ISTA (g) and FISTA (b) and their sign support (+)')
    plt.plot(x_0, 'r-')
    plt.plot(x_ista, 'g-')
    plt.plot(x_fista, 'b-')    
    plt.plot(x_ista_step, 'g+')
    plt.plot(x_fista_step, 'b+')
    
    plt.subplot(223)
    plt.title('True signal (r) and recovered signals of ISTA (g) and FISTA (b)')
    plt.plot(obs, 'r')
    plt.plot(np.matmul(dict, x_ista), 'g-')
    plt.plot(np.matmul(dict, x_fista), 'b-')    
    
    plt.show()


