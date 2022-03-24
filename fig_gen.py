import examples as examples
import utils as utils
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ISTA as ISTA
import LISTA as LISTA

##Compare all at once training and layer by layer training w/wo weight decay
def fig_gen_compare_aao_lbl_wd(cv = False, lambd = 0.1, dict = None, list_x_0 = None,  sparsity = 10, n_cols = 200, n_rows = 100, train_size = 500, test_size = 100, batch_size = None, epochs = None):
    
    dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)
    list_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(train_size)])

    (history1, avg_error1, single_x_0,single_x_hat_trained) = examples.Test_example_LISTA_16_Layer(weight_decay = True, dict = dict, list_x_0 = list_x_0, train_size=train_size, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)
    (history0, avg_error0, single_x_0,single_x_hat_trained) = examples.Test_example_LISTA_16_Layer(weight_decay = False, dict = dict, list_x_0 = list_x_0, train_size=train_size, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)
    (history2, avg_error2, single_x_0,single_x_hat_trained) = examples.Test_example_LISTA_16_Layer(weight_decay = False, AAO = True, dict = dict, list_x_0 = list_x_0, train_size=train_size, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)

    
    plt.figure()
    plt.title('mse through learning')
    plt.plot(history1['loss'], label='weighted loss')
    plt.plot(history1['output_1_loss'], label='1st layer output (WD)')
    plt.plot(history1['output_4_loss'], label='4th layer output (WD)')
    plt.plot(history1['output_8_loss'], label='8th layer output (WD)')
    plt.plot(history1['output_12_loss'], label='12th layer output (WD)')
    plt.plot(history1['output_16_loss'], label='16th layer output (WD)')    
    plt.plot(history0['loss'], label='weighted loss (No WD)', linestyle = '--')
    plt.plot(history0['output_1_loss'], label='1st layer output (No WD)'    , linestyle = '--')
    plt.plot(history0['output_4_loss'], label='4th layer output (No WD)'    , linestyle = '--')
    plt.plot(history0['output_8_loss'], label='8th layer output (No WD)'    , linestyle = '--')
    plt.plot(history0['output_12_loss'], label='12th layer output (No WD)'  , linestyle = '--')
    plt.plot(history0['output_16_loss'], label='16th layer output (No WD)'  , linestyle = '--')
    plt.plot(history2['loss'], label='weighted loss (AAO)'  , linestyle='-')
    plt.plot(history2['output_1_loss'], label='1st layer output (AAO)', linestyle='-')
    plt.plot(history2['output_4_loss'], label='4th layer output (AAO)', linestyle='-')
    plt.plot(history2['output_8_loss'], label='8th layer output (AAO)', linestyle='-')
    plt.plot(history2['output_12_loss'], label='12th layer output(AAO)', linestyle='-')
    plt.plot(history2['output_16_loss'], label='16th layer output(AAO)', linestyle='-') 
    plt.legend()
    plt.show()
    plt.figure()
    plt.title('mse through layers on test data')
    plt.plot(avg_error0, label="average error (No WD)")
    plt.plot(avg_error1, label="average error (WD)")
    plt.plot(avg_error2, label="average error (AAO)")
    plt.legend()
    plt.show()



##Compare all at once training and layer by layer training
def fig_gen_compare_aao_lbl(cv = False, lambd = 0.1, dict = None, list_x_0 = None,  sparsity = 10, n_cols = 200, n_rows = 100, train_size_aao = 500, train_size_lbl = 500, test_size = 100, batch_size = None, epochs = None):
    
    if dict is None:
            dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)
    if train_size_aao == train_size_lbl:
        if list_x_0 is None:
            list_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(train_size_aao)])
    list_test_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(test_size)])

    #(history0, avg_error0, single_x_0,single_x_hat_trained_LBL, eig_max_list, theta_list, model_LBL) = examples.Test_example_LISTA_16_Layer(weight_decay = False, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)
    #(history1, avg_error2, single_x_0,single_x_hat_trained_AAO, eig_max_list, theta_list, model_AAO) = examples.Test_example_LISTA_16_Layer(weight_decay = False, AAO = True, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)

    (history0, avg_error0, single_x_0,single_x_hat_trained_LBL, eig_max_list_LBL, theta_list_LBL, model_LBL) = examples.Test_example_LISTA_16_Layer(weight_decay = False, compute_par_decay=True, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size_lbl, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)
    (history1, avg_error2, single_x_0,single_x_hat_trained_AAO, eig_max_list_AAO, theta_list_AAO, model_AAO) = examples.Test_example_LISTA_16_Layer(weight_decay = False, compute_par_decay = True, AAO = True, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size_aao, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)
    
    plt.figure()
    plt.title('mse through learning Layer by Layer training (LBL) and All At Once (AAO)')
    plt.plot(history0['loss'], label='weighted loss')
    plt.plot(history0['output_1_loss'], label='1st layer output LBL')
    plt.plot(history0['output_4_loss'], label='4th layer output LBL')
    plt.plot(history0['output_8_loss'], label='8th layer output LBL')
    plt.plot(history0['output_12_loss'], label='12th layer output LBL')
    plt.plot(history0['output_16_loss'], label='16th layer output LBL')    
    plt.plot(history1['output_1_loss'], label='1st layer output AAO'    , linestyle = '--')
    plt.plot(history1['output_4_loss'], label='4th layer output AAO'    , linestyle = '--')
    plt.plot(history1['output_8_loss'], label='8th layer output AAO'    , linestyle = '--')
    plt.plot(history1['output_12_loss'], label='12th layer output AAO'  , linestyle = '--')
    plt.plot(history1['output_16_loss'], label='16th layer output AAO'  , linestyle = '--')
    plt.legend()
    plt.show()
    plt.figure()
    plt.title('mse through layers on test data')
    plt.plot(avg_error0, label="average error Layer by layer")
    plt.plot(avg_error2, label="average error All at once")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('True coefficients, and coefficients recovered by untrained and trained LISTA')
    plt.plot(single_x_0, label='true coefficients')
    plt.plot(single_x_hat_trained_LBL[0], label='LISTA 1st layer output (LBL)', linestyle='--')
    plt.plot(single_x_hat_trained_AAO[0], label='LISTA 1st layer output (AAO)')
    plt.plot(single_x_hat_trained_LBL[11], label='trained LISTA 8th layer output (LBL)', linestyle='--')
    plt.plot(single_x_hat_trained_AAO[11], label='trained LISTA 8th layer output (AAO)')
    plt.plot(single_x_hat_trained_LBL[15], label='trained LISTA 16th layer output (LBL)', linestyle='--')
    plt.plot(single_x_hat_trained_AAO[15], label='trained LISTA 16th layer output (AAO)')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('decay of theta and validation of necessary condition')
    plt.plot(eig_max_list_AAO, label='operator norm (AAO)')
    plt.plot(eig_max_list_LBL, label='operator norm (LBL)')
    plt.plot(theta_list_AAO, label='value of theta (AAO)')
    plt.plot(theta_list_LBL, label='value of theta (LBL)')
    plt.legend()
    plt.show()

##Compare layer by layer training with fixed layer step
def fig_gen_compare_lbl_no_step(cv = False, lambd = 0.1, dict = None, list_x_0 = None,  sparsity = 10, n_cols = 200, n_rows = 100, train_size = 500, test_size = 100, batch_size = None, epochs = None):
    
    dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)
    list_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(train_size)])
    list_test_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(test_size)])

    (history0, avg_error0, single_x_0,single_x_hat_trained, eig_max_list, theta_list) = examples.Test_example_LISTA_16_Layer(weight_decay = False, double_pass = True, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)
    (history1, avg_error2, single_x_0,single_x_hat_trained, eig_max_list, theta_list) = examples.Test_example_LISTA_16_Layer(weight_decay = False, double_pass = False, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)

    
    plt.figure()
    plt.title('mse through layer by layer learning with and without fixed step')
    plt.plot(history0['loss'], label='weighted loss')
    plt.plot(history0['output_1_loss'], label='1st layer output with step')
    plt.plot(history0['output_4_loss'], label='4th layer output with step')
    plt.plot(history0['output_8_loss'], label='8th layer output with step')
    plt.plot(history0['output_12_loss'], label='12th layer output with step')
    plt.plot(history0['output_16_loss'], label='16th layer output with step')    
    plt.plot(history1['output_1_loss'], label='1st layer output without step'    , linestyle = '--')
    plt.plot(history1['output_4_loss'], label='4th layer output without step'    , linestyle = '--')
    plt.plot(history1['output_8_loss'], label='8th layer output without step'    , linestyle = '--')
    plt.plot(history1['output_12_loss'], label='12th layer output without step'  , linestyle = '--')
    plt.plot(history1['output_16_loss'], label='16th layer output without step'  , linestyle = '--')
    plt.legend()
    plt.show()
    plt.figure()
    plt.title('mse through layers on test data')
    plt.plot(avg_error0, label="average error with step")
    plt.plot(avg_error2, label="average error without step")
    plt.legend()
    plt.show()
    

def fig_gen_compare_LISTA_ISTA(cv = False, lambd = 0.1, dict = None, list_x_0 = None,  sparsity = 10, n_cols = 200, n_rows = 100, train_size = 500, test_size = 100, batch_size = None, epochs = 1, AAO = True, ):
    
    dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)
    list_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(train_size)])
    list_test_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(test_size)])
    list_obs = [tf.linalg.matvec(dict, list_x_0[i]) for i in range(len(list_x_0))]

    
    (history0, avg_error_LISTA, single_x_0,x_hat_LISTA, eig_l, thet_l, model) = examples.Test_example_LISTA_16_Layer(AAO = AAO, CP=False, weight_decay = False, double_pass = False, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)
    (history0, avg_error_LISTA_CP, single_x_0,x_hat_LISTA_CP, eig_l, thet_l, model) = examples.Test_example_LISTA_16_Layer(AAO = AAO, CP=True, weight_decay = False, double_pass = False, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)

    avg_error_FISTA = examples.Test_example_FISTA_avg(show_plots = True, cv = False, lambd = 0.1, dict = dict, list_obs=list_obs, list_x_0=list_x_0, sparsity = 10, n_cols = n_cols, n_rows = n_rows, k_max = max(16*epochs, 3*sparsity))
    avg_error_ISTA = examples.Test_example_ISTA_avg(show_plots = True, cv = False, lambd = 0.1, dict = dict, list_obs=list_obs, list_x_0=list_x_0, sparsity = 10, n_cols = n_cols, n_rows = n_rows, k_max = max(16*epochs, 3*sparsity))
    avg_error_FISTA_Im = examples.Test_example_FISTA_avg_Im(show_plots = True, cv = False, lambd = 0.1, dict = dict, list_obs=list_obs, list_x_0=list_x_0, sparsity = 10, n_cols = n_cols, n_rows = n_rows, k_max = max(16*epochs, 3*sparsity))
    avg_error_ISTA_Im = examples.Test_example_ISTA_avg_Im(show_plots = True, cv = False, lambd = 0.1, dict = dict, list_obs=list_obs, list_x_0=list_x_0, sparsity = 10, n_cols = n_cols, n_rows = n_rows, k_max = max(16*epochs, 3*sparsity))
    print(avg_error_ISTA)
    plt.figure()
    plt.title('mse through ' + str(max(16*epochs,3*sparsity)) + ' iterations for ISTA and FISTA, and through layers for LISTA')
    plt.plot(avg_error_ISTA, label = 'ISTA error')
    plt.plot(avg_error_FISTA, label = 'FISTA error')
    plt.plot(avg_error_ISTA_Im, label = 'ISTA data-fidelity error')
    plt.plot(avg_error_FISTA_Im, label = 'FISTA data-fidelity error')
    plt.plot(avg_error_LISTA, label = 'LISTA error')
    plt.plot(avg_error_LISTA_CP, label = 'LISTA CP error')
    plt.legend()
    plt.show()

    max_eig = max(np.linalg.eigvals(np.matmul(dict.transpose(), dict)))

    x_hat_ISTA, res_ISTA = ISTA.ISTA_solver_true_res(A = dict, obs = np.matmul(dict, single_x_0), true_val=single_x_0, k_max =100, L = max_eig, lambd=0.1)
    x_hat_FISTA, res_FISTA = ISTA.FISTA_solver_true_res(A = dict, obs = np.matmul(dict, single_x_0), true_val=single_x_0, k_max =100, L = max_eig, lambd=0.1)
    print('On single sample:')
    print('ISTA error : ' + str(res_ISTA))
    print('FISTA error : ' + str(res_FISTA))
    plt.figure()
    plt.title('Comparison of ISTA, FISTA, LISTA on a single sample')
    plt.plot(single_x_0, label='True coefficients')
    plt.plot(x_hat_ISTA, label='ISTA coefficients')
    plt.plot(x_hat_FISTA, label='FISTA coefficients')
    plt.plot(x_hat_LISTA[-1], label='LISTA coefficients')
    plt.plot(x_hat_LISTA_CP[-1], label='LISTA CP coefficients')
    plt.legend()
    plt.show()

 ##Compare all at once training and layer by layer training
def fig_gen_compare_aao_lbl_cp(cv = False, lambd = 0.1, dict = None, list_x_0 = None,  sparsity = 10, n_cols = 200, n_rows = 100, train_size_aao = 500, train_size_lbl = 500, test_size = 100, batch_size = None, epochs = None):
    
    if dict is None:
            dict = utils.Generate_gaussian_dictionary(n_cols = n_cols, n_rows = n_rows, se = 1)
    if train_size_aao == train_size_lbl:
        if list_x_0 is None:
            list_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(train_size_aao)])
    list_test_x_0 = np.array([tf.convert_to_tensor(utils.Create_sparse_x(len(dict[0]), sparsity)) for i in range(test_size)])

    #(history0, avg_error0, single_x_0,single_x_hat_trained_LBL, eig_max_list, theta_list, model_LBL) = examples.Test_example_LISTA_16_Layer(weight_decay = False, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)
    #(history1, avg_error2, single_x_0,single_x_hat_trained_AAO, eig_max_list, theta_list, model_AAO) = examples.Test_example_LISTA_16_Layer(weight_decay = False, AAO = True, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)

    (history0, avg_error0, single_x_0,single_x_hat_trained_LBL, eig_max_list_LBL, theta_list_LBL, model_LBL) = examples.Test_example_LISTA_16_Layer(weight_decay = False, compute_par_decay=True, double_pass=False, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size_lbl, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)
    (history1, avg_error2, single_x_0,single_x_hat_trained_AAO, eig_max_list_AAO, theta_list_AAO, model_AAO) = examples.Test_example_LISTA_16_Layer(weight_decay = False, compute_par_decay = True, AAO = True, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size_aao, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)
    (history0_CP, avg_error0_CP, single_x_0,single_x_hat_trained_LBL_CP, eig_max_list_LBL_CP, theta_list_LBL_CP, model_LBL) = examples.Test_example_LISTA_16_Layer(CP = True, weight_decay = False, compute_par_decay=True, double_pass=False, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size_lbl, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)
    (history1_CP, avg_error2_CP, single_x_0,single_x_hat_trained_AAO_CP, eig_max_list_AAO_CP, theta_list_AAO_CP, model_AAO) = examples.Test_example_LISTA_16_Layer(CP = True,weight_decay = False, compute_par_decay = True, AAO = True, dict = dict, list_x_0 = list_x_0, list_test_x_0 = list_test_x_0, train_size=train_size_aao, n_cols=n_cols, n_rows=n_rows, batch_size = batch_size, epochs = epochs)
      
    plt.figure()
    plt.title('mse through learning Layer by Layer training (LBL), with or without partial coupling (CP)')
    plt.plot(history0['loss'], label='weighted loss LBL')
    plt.plot(history0_CP['loss'], label='weighted loss (CP)')

    plt.plot(history0['output_16_loss'], label='16th layer output LBL')    
    plt.plot(history0_CP['output_16_loss'], label='16th layer output LBL (CP)')    
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('mse through layers on test data')
    plt.plot(avg_error0, label="average error Layer by layer")
    plt.plot(avg_error2, label="average error All at once")    
    plt.plot(avg_error0_CP, label="average error Layer by layer (CP)")
    plt.plot(avg_error2_CP, label="average error All at once (CP)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('True coefficients, and coefficients recovered by untrained and trained LISTA')
    plt.plot(single_x_0, label='true coefficients')
    plt.plot(single_x_hat_trained_LBL[15], label='trained LISTA 16th layer output (LBL)', linestyle='--')
    plt.plot(single_x_hat_trained_AAO[15], label='trained LISTA 16th layer output (AAO)')   
    plt.plot(single_x_hat_trained_LBL_CP[15], label='trained LISTA 16th layer output (LBL-CP)', linestyle='--')
    plt.plot(single_x_hat_trained_AAO_CP[15], label='trained LISTA 16th layer output (AAO-CP)')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('validation of necessary condition')
    plt.plot(eig_max_list_AAO, label='operator norm (AAO)')
    plt.plot(eig_max_list_LBL, label='operator norm (LBL)')
    #plt.plot(eig_max_list_AAO_CP, label='operator norm (AAO)')
    #plt.plot(eig_max_list_LBL_CP, label='operator norm (LBL)')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Decay of theta')
    plt.plot(theta_list_AAO, label='value of theta (AAO)')
    plt.plot(theta_list_LBL, label='value of theta (LBL)')    
    plt.plot(theta_list_AAO_CP, label='value of theta (AAO-CP)')
    plt.plot(theta_list_LBL_CP, label='value of theta (LBL)')
    plt.legend()
    plt.show()

