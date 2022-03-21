import utils as utils
from ISTA import ISTA_solver, FISTA_solver, FISTA_solver_true_res, ISTA_solver_true_res, ISTA_solver_true_res_Im, FISTA_solver_true_res_Im
import LISTA as lista
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

def Test_example_FISTA_avg(show_plots = False, cv = False, test_cv =10, lambd = 0.1, dict = None, list_obs = None, list_x_0 = None, sparsity = 10, n_cols = 200, n_rows = 100, k_max = 100):
    if dict is None:
        dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)
    if list_obs is None:
        x_0 = utils.Create_sparse_x(len(dict[0]), sparsity)
        list_obs = [tf.linalg.matvec(dict, list_x_0[i]) for i in range(len(list_x_0))]

    max_eig = max(np.linalg.eigvals(np.matmul(dict.transpose(), dict)))

    if cv:
        FISTA_solver_fixed_param = lambda y, lambd, A : FISTA_solver(A = A, obs = y, k_max = k_max, L=max_eig, lambd = lambd)
        hyperparams_list = np.linspace(0,2,50)
        cv_lambda = utils.Cross_validation_list_obs(list_obs=list_obs[0:test_cv], list_x_0=list_x_0[0:test_cv], dict = dict, solver = FISTA_solver_fixed_param, nfolds=2, hyperparameters=hyperparams_list, plot=True)
        print("In Test_example_FISTA_avg(): lambda selected by cross-validation : " + str(cv_lambda))
        lambd = cv_lambda
    RSS = np.zeros(k_max)
    for k in range(len(list_obs)):
        (x_hat,res) = FISTA_solver_true_res(A = dict, true_val = list_x_0[k], obs = list_obs[k], k_max=k_max, L=max_eig, lambd = lambd)
        RSS += res
    for k in range(len(RSS)):
        RSS[k] *= 1.0/ float( len(list_obs))
    return RSS
def Test_example_FISTA_avg_Im(show_plots = False, cv = False, test_cv =10, lambd = 0.1, dict = None, list_obs = None, list_x_0 = None, sparsity = 10, n_cols = 200, n_rows = 100, k_max = 100):
    if dict is None:
        dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)
    if list_obs is None:
        x_0 = utils.Create_sparse_x(len(dict[0]), sparsity)
        list_obs = [tf.linalg.matvec(dict, list_x_0[i]) for i in range(len(list_x_0))]

    max_eig = max(np.linalg.eigvals(np.matmul(dict.transpose(), dict)))

    if cv:
        FISTA_solver_fixed_param = lambda y, lambd, A : FISTA_solver(A = A, obs = y, k_max = k_max, L=max_eig, lambd = lambd)
        hyperparams_list = np.linspace(0,2,50)
        cv_lambda = utils.Cross_validation_list_obs(list_obs=list_obs[0:test_cv], list_x_0=list_x_0[0:test_cv], dict = dict, solver = FISTA_solver_fixed_param, nfolds=2, hyperparameters=hyperparams_list, plot=True)
        print("In Test_example_FISTA_avg(): lambda selected by cross-validation : " + str(cv_lambda))
        lambd = cv_lambda
    RSS = np.zeros(k_max)
    for k in range(len(list_obs)):
        (x_hat,res) = FISTA_solver_true_res_Im(A = dict, true_val = list_x_0[k], obs = list_obs[k], k_max=k_max, L=max_eig, lambd = lambd)
        RSS += res
    for k in range(len(RSS)):
        RSS[k] *= 1.0/ float( len(list_obs))
    return RSS
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

def Test_example_ISTA_avg(show_plots = False, cv = False, test_cv =10, lambd = 0.1, dict = None, list_obs = None, list_x_0 = None, sparsity = 10, n_cols = 200, n_rows = 100, k_max = 100):
    if dict is None:
        dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)
    if list_obs is None:
        x_0 = utils.Create_sparse_x(len(dict[0]), sparsity)
        list_obs = [tf.linalg.matvec(dict, list_x_0[i]) for i in range(len(list_x_0))]

    max_eig = max(np.linalg.eigvals(np.matmul(dict.transpose(), dict)))

    if cv:
        FISTA_solver_fixed_param = lambda y, lambd, A : ISTA_solver(A = A, obs = y, k_max = k_max, L=max_eig, lambd = lambd)
        hyperparams_list = np.linspace(0,2,50)
        cv_lambda = utils.Cross_validation_list_obs(list_obs=list_obs[0:test_cv], list_x_0=list_x_0[0:test_cv], dict = dict, solver = FISTA_solver_fixed_param, nfolds=2, hyperparameters=hyperparams_list, plot=True)
        print("In Test_example_FISTA_avg(): lambda selected by cross-validation : " + str(cv_lambda))
        lambd = cv_lambda
    RSS = np.zeros(k_max)
    for k in range(len(list_obs)):
        (x_hat,res) = ISTA_solver_true_res(A = dict, true_val = list_x_0[k], obs = list_obs[k], k_max=k_max, L=max_eig, lambd = lambd)
        RSS += res
    for k in range(len(RSS)):
        RSS[k] *= 1.0/ float( len(list_obs))
    return RSS

def Test_example_ISTA_avg_Im(show_plots = False, cv = False, test_cv =10, lambd = 0.1, dict = None, list_obs = None, list_x_0 = None, sparsity = 10, n_cols = 200, n_rows = 100, k_max = 100):
    if dict is None:
        dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)
    if list_obs is None:
        x_0 = utils.Create_sparse_x(len(dict[0]), sparsity)
        list_obs = [tf.linalg.matvec(dict, list_x_0[i]) for i in range(len(list_x_0))]

    max_eig = max(np.linalg.eigvals(np.matmul(dict.transpose(), dict)))

    if cv:
        FISTA_solver_fixed_param = lambda y, lambd, A : ISTA_solver(A = A, obs = y, k_max = k_max, L=max_eig, lambd = lambd)
        hyperparams_list = np.linspace(0,2,50)
        cv_lambda = utils.Cross_validation_list_obs(list_obs=list_obs[0:test_cv], list_x_0=list_x_0[0:test_cv], dict = dict, solver = FISTA_solver_fixed_param, nfolds=2, hyperparameters=hyperparams_list, plot=True)
        print("In Test_example_FISTA_avg(): lambda selected by cross-validation : " + str(cv_lambda))
        lambd = cv_lambda
    RSS = np.zeros(k_max)
    for k in range(len(list_obs)):
        (x_hat,res) = ISTA_solver_true_res_Im(A = dict, true_val = list_x_0[k], obs = list_obs[k], k_max=k_max, L=max_eig, lambd = lambd)
        RSS += res
    for k in range(len(RSS)):
        RSS[k] *= 1.0/ float( len(list_obs))
    return RSS


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



##Test_example_LISTA_1_Layer: setups a 1 layer LISTA, trains it and returns mse over some test set
#All parameters are optional
#show_plots: boolean value on whether or not to display various plots
#dict: fixed dictionary
#train_size: number of samples to train LISTA
#test_size: number of samples to test the trained LISTA

def Test_example_LISTA_1_Layer(show_plots = False, cv = False, lambd = 0.1, dict = None,  sparsity = 10, n_cols = 200, n_rows = 100, train_size = 500, test_size = 100, batch_size = None, epochs = None):
    if dict is None:
        dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)

    #Gen model
    model = lista.setup_LISTA_1_Layer(signal_dim = n_rows, sol_dim = n_cols, dict=dict)

    #We simulate a single signal that we will use to show how LISTA evolves before and after training on a specific example
    #This signal is not seen during training
    if show_plots:
        single_x_0 = tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) 
        single_obs = tf.linalg.matvec(dict, single_x_0)
        single_x_hat_untrained = model(single_obs)

    #Gen true solutions and observations to train the model
    list_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(train_size)])
    list_obs = [tf.linalg.matvec(dict, list_x_0[i]) for i in range(len(list_x_0))]

    #Train model
    history = lista.train_LISTA_1_Layer(model=model, array_obs=list_obs, array_sols = list_x_0, batch_size = batch_size, epochs=epochs)
    print(history.history['loss'])

    #Test model
    list_test_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(test_size)])
    list_test_obs = [tf.linalg.matvec(dict, list_x_0[i]) for i in range(len(list_test_x_0))]
    list_test_x_hat = [model(input_signal = list_test_obs[i]) for i in range(len(list_test_obs))]
    avg_error =  tf.reduce_mean(tf.square(list_test_x_0 - list_test_x_hat))
    print("average error when testing trained 1 layer LISTA : " + str(avg_error.numpy()))

    if show_plots:
        plt.figure()
        plt.title('mse through learning')
        plt.plot(history.history['loss'])
        plt.show()


        single_x_hat_trained = model(single_obs)
        plt.figure()
        plt.title('True coefficients, and coefficients recovered by untrained and trained LISTA')
        plt.plot(single_x_0, label='true coefficients')
        plt.plot(single_x_hat_untrained, label='untrained LISTA')
        plt.plot(single_x_hat_trained, label='trained LISTA')
        plt.legend()
        plt.show()


##Test_example_LISTA_4_Layer: setups a 4 layer LISTA, trains it and returns mse over some test set
#All parameters are optional
#show_plots: boolean value on whether or not to display various plots
#dict: fixed dictionary
#train_size: number of samples to train LISTA
#test_size: number of samples to test the trained LISTA

def Test_example_LISTA_4_Layer(show_plots = False, cv = False, lambd = 0.1, dict = None,  sparsity = 10, n_cols = 200, n_rows = 100, train_size = 500, test_size = 100, batch_size = None, epochs = None):
    if dict is None:
        dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)

    #Gen model
    model = lista.setup_LISTA_4_Layer(signal_dim = n_rows, sol_dim = n_cols, dict=dict)

    #We simulate a single signal that we will use to show how LISTA evolves before and after training on a specific example
    #This signal is not seen during training
    if show_plots:
        single_x_0 = tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) 
        single_obs = tf.linalg.matvec(dict, single_x_0)
        single_x_hat_untrained = model(single_obs)[-1]

    #Gen true solutions and observations to train the model
    list_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(train_size)])
    list_obs = [tf.linalg.matvec(dict, list_x_0[i]) for i in range(len(list_x_0))]

    #Train model
    history= lista.train_LISTA_4_Layer(model=model, array_obs=list_obs, array_sols = list_x_0, batch_size = batch_size, epochs=epochs)
    print(history)
    #Test model
    list_test_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(test_size)])
    list_test_obs = [tf.linalg.matvec(dict, list_test_x_0[i]) for i in range(len(list_test_x_0))] ####test ?
    list_test_x_hat = [model(input_signal = list_test_obs[i])[-1] for i in range(len(list_test_obs))]
    
    avg_error =  tf.reduce_mean(tf.square(list_test_x_0 - list_test_x_hat))
    print("average error when testing trained 4 layer LISTA : " + str(avg_error.numpy()))

    if show_plots:
        plt.figure()
        plt.title('mse through learning')
        plt.plot(history['loss'], label='weighted loss')
        plt.plot(history['output_1_loss'], label='1st layer output')
        plt.plot(history['output_2_loss'], label='2nd layer output')
        plt.plot(history['output_3_loss'], label='3rd layer output')
        plt.plot(history['output_4_loss'], label='4th layer output')
        plt.legend()
        plt.show()


        single_x_hat_trained = model(single_obs)
        plt.figure()
        plt.title('True coefficients, and coefficients recovered by untrained and trained LISTA')
        plt.plot(single_x_0, label='true coefficients')
        plt.plot(single_x_hat_untrained, label='untrained LISTA')
        plt.plot(single_x_hat_trained[0], label='trained LISTA 1st layer output')
        plt.plot(single_x_hat_trained[1], label='trained LISTA 2nd layer output')
        plt.plot(single_x_hat_trained[2], label='trained LISTA 3rd layer output')
        plt.plot(single_x_hat_trained[3], label='trained LISTA 4th layer output')
        plt.legend()
        plt.show()



##Test_example_LISTA_4_Layer: setups a 4 layer LISTA, trains it and returns mse over some test set
#All parameters are optional
#show_plots: boolean value on whether or not to display various plots
#dict: fixed dictionary
#train_size: number of samples to train LISTA
#test_size: number of samples to test the trained LISTA

def Test_example_LISTA_16_Layer(show_plots = False, cv = False, lambd = 0.1, dict = None, list_x_0 = None, list_test_x_0 = None,  sparsity = 10, n_cols = 200, n_rows = 100, train_size = 500, test_size = 100, batch_size = None, epochs = None, weight_decay = False, AAO = False, double_pass = True):
    L=16
    if dict is None:
        dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)

    #Gen model
    model = lista.setup_LISTA_L_Layer(signal_dim = n_rows, sol_dim = n_cols, dict=dict, L=16)

    #We simulate a single signal that we will use to show how LISTA evolves before and after training on a specific example
    #This signal is not seen during training
    single_x_0 = tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) 
    single_obs = tf.linalg.matvec(dict, single_x_0)
    single_x_hat_untrained = model(single_obs)[-1]      ##-1 is to get output from last layer

    #Gen true solutions and observations to train the model
    if list_x_0 is None:
        list_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(train_size)])
    list_obs = [tf.linalg.matvec(dict, list_x_0[i]) for i in range(len(list_x_0))]

    #Train model
    history= lista.train_LISTA_L_Layer(model=model, array_obs=list_obs, array_sols = list_x_0, batch_size = batch_size, epochs=epochs, L=16, weight_decay = weight_decay, AAO = AAO, double_pass=double_pass)
    print(history)
    #Test model
    if list_test_x_0 is None:
        list_test_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(test_size)])
    list_test_obs = [tf.linalg.matvec(dict, list_test_x_0[i]) for i in range(len(list_test_x_0))]
    list_test_x_hat = []
    for j in range(L):
        list_test_x_hat.append([model(input_signal = list_test_obs[i])[j] for i in range(len(list_test_obs))])
    avg_error = []
    for j in range(L):
        avg_error.append(tf.reduce_mean(tf.square(list_test_x_0 - list_test_x_hat[j])))
    print("average error when testing trained 16 layer LISTA : " + str(avg_error[-1].numpy()))

    single_x_hat_trained = model(single_obs)

    if show_plots:
        plt.figure()
        plt.title('mse through learning')
        plt.plot(history['loss'], label='weighted loss')
        plt.plot(history['output_1_loss'], label='1st layer output')
        plt.plot(history['output_4_loss'], label='4th layer output')
        plt.plot(history['output_8_loss'], label='8th layer output')
        plt.plot(history['output_12_loss'], label='12th layer output')
        plt.plot(history['output_16_loss'], label='16th layer output')
        plt.legend()
        plt.show()

        plt.figure()
        plt.title('mse through layers on test data')
        plt.plot(avg_error, label="average error")
        plt.legend()
        plt.show()

        plt.figure()
        plt.title('True coefficients, and coefficients recovered by untrained and trained LISTA')
        plt.plot(single_x_0, label='true coefficients')
        plt.plot(single_x_hat_untrained, label='untrained LISTA', linestyle="-")
        plt.plot(single_x_hat_trained[0], label='trained LISTA 1st layer output')
        plt.plot(single_x_hat_trained[3], label='trained LISTA 4th layer output')
        plt.plot(single_x_hat_trained[7], label='trained LISTA 8th layer output')
        plt.plot(single_x_hat_trained[11], label='trained LISTA 12th layer output')
        plt.plot(single_x_hat_trained[15], label='trained LISTA 16th layer output')
        plt.legend()
        plt.show()

    return history, avg_error, single_x_0, single_x_hat_trained















def Test_example_LISTA_rec_Layer(show_plots = False, depth = 2, cv = False, lambd = 0.1, dict = None,  sparsity = 10, n_cols = 200, n_rows = 100, train_size = 500, test_size = 100, batch_size = None, epochs = None):
    if dict is None:
        dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)

    #Gen model
    model = lista.setup_LISTA_rec_Layer(signal_dim = n_rows, sol_dim = n_cols, network_depth=depth)
    print("Generated a recurrent network model of LISTA of depth :" + str(model.depth))

    #We simulate a single signal that we will use to show how LISTA evolves before and after training on a specific example
    #This signal is not seen during training
    if show_plots:
        single_x_0 = tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) 
        single_obs = tf.linalg.matvec(dict, single_x_0)
        single_x_hat_untrained = model(single_obs)

    #Gen true solutions and observations to train the model
    list_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(train_size)])
    list_obs = [tf.linalg.matvec(dict, list_x_0[i]) for i in range(len(list_x_0))]

    #Train model
    print("About to start training LISTA_rec")
    history = lista.train_LISTA_rec_Layer(model=model, array_obs=list_obs, array_sols = list_x_0, batch_size = batch_size, epochs=epochs)
    print(history.history['loss'])

    #Test model
    list_test_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(test_size)])
    list_test_obs = [tf.linalg.matvec(dict, list_x_0[i]) for i in range(len(list_test_x_0))]
    list_test_x_hat = [model(input_signal = list_test_obs[i]) for i in range(len(list_test_obs))]
    
    
    avg_error =  tf.reduce_mean(tf.square(list_test_x_0 - list_test_x_hat))
    print("average error when testing trained recurrent LISTA : " + str(avg_error.numpy()))

    if show_plots:
        plt.figure()
        plt.title('mse through learning')
        plt.plot(history.history['loss'])
        plt.show()


        single_x_hat_trained = model(single_obs)
        plt.figure()
        plt.title('True coefficients, and coefficients recovered by untrained and trained recurrent LISTA')
        plt.plot(single_x_0, label='true coefficients')
        plt.plot(single_x_hat_untrained, label='untrained LISTA')
        plt.plot(single_x_hat_trained, label='trained LISTA')
        plt.legend()
        plt.show()

def Test_example_LISTA_iter(show_plots = False, depth = 2, cv = False, lambd = 0.1, dict = None,  sparsity = 10, n_cols = 200, n_rows = 100, train_size = 500, test_size = 100, batch_size = None, epochs = None):
    if dict is None:
        dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)

    #Gen model
    model = lista.setup_LISTA_iter(signal_dim = n_rows, sol_dim = n_cols, network_depth=depth)
    print("Generated a iterative network model of LISTA of depth :" + str(model.depth))

    #We simulate a single signal that we will use to show how LISTA evolves before and after training on a specific example
    #This signal is not seen during training
    if show_plots:
        single_x_0 = tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) 
        single_obs = tf.linalg.matvec(dict, single_x_0)
        single_x_hat_untrained = model(single_obs)

    #Gen true solutions and observations to train the model
    list_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(train_size)])
    list_obs = [tf.linalg.matvec(dict, list_x_0[i]) for i in range(len(list_x_0))]

    #Train model
    print("About to start training LISTA_iter")
    loss = lista.train_LISTA_iter(model=model, array_obs=list_obs, array_sols = list_x_0, batch_size = batch_size, epochs=epochs)
    print(loss)

    #Test model
    list_test_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(test_size)])
    list_test_obs = [tf.linalg.matvec(dict, list_x_0[i]) for i in range(len(list_test_x_0))]
    list_test_x_hat = [model(input_signal = list_test_obs[i]) for i in range(len(list_test_obs))]
    avg_error =  tf.reduce_mean(tf.square(list_test_x_0 - list_test_x_hat))
    print("average error when testing trained iterative LISTA : " + str(avg_error.numpy()))

    if show_plots:
        plt.figure()
        plt.title('mse through learning')
        plt.plot(loss)
        plt.show()


        single_x_hat_trained = model(single_obs)
        plt.figure()
        plt.title('True coefficients, and coefficients recovered by untrained and trained iterative LISTA')
        plt.plot(single_x_0, label='true coefficients')
        plt.plot(single_x_hat_untrained, label='untrained LISTA')
        plt.plot(single_x_hat_trained, label='trained LISTA')
        plt.legend()
        plt.show()


