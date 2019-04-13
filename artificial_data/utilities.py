import os
import tensorflow as tf
import numpy as np
from IPython.display import clear_output, Image, display, HTML
import scipy.io
from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
import tensorflow.contrib.slim as slim
from mlxtend.evaluate import mcnemar
import csv
from mlxtend.evaluate import mcnemar_table
import argparse

### define all the variables 
def get_arguments():

    parser = argparse.ArgumentParser(description='Three step optimization')
    #Model parameters
    parser.add_argument('--epsilon', type=float, default="1.1", help="epsilon to filter the augmented data")
    parser.add_argument('--reg_param1', type=float, default="100", help="regularizer parameter for y_y1")
    parser.add_argument('--reg_param2', type=float, default="1", help="regularizer parameter for fx1_y")
    parser.add_argument('--sigma', type=float, default="0.02", help="variance of added guassian noise")
    #network1 parameter
    parser.add_argument('--net1_h1', type=int, default="100", help="main network hidden layer 1 size")
    parser.add_argument('--net1_h2', type=int, default="50", help="main network hidden layer 2 size")
    #parser.add_argument('--sigma', type=float, default="0.02", help="gaussian noise stadard deviation size")
        
    #input output paths
    parser.add_argument('--data_name', type=str, help='name of the data we are calling')
    parser.add_argument('--data_dir', type=str, default='/work/amina/RawM_level_datacleaning2017_6_1/model/StabilityModel/Tensorflow_code/11_27_final/Test_on_text/ICDM/artificial_data/', help='directory which contains input data')
    parser.add_argument('--output_dir', type=str, default='./output/', help='directory for output of logs for tensorboard')
    parser.add_argument('--model_name', type=str, default='ICDM_fx1_y', help="the name of the model")
	#optimization parameters  
    parser.add_argument('--batch_size', type=int, default="20", help="batchsize (default: 50)")                  
    parser.add_argument('--epochs', type=int, default="2000", help="optimization 1 epoch number (default: 36)")
    parser.add_argument('--Save_every',  type=int, default="100", help='Save the train and validation loss at after __ update (default: 100)')
    parser.add_argument('--lr', type=float, default="1e-4", help='learning rate (default: 1e-3)')                    

	#early stopping parameters
    parser.add_argument('--start_early_stop', type=int, default="1900", help='Activate early stopping after update_num (default:1900)')                           
    parser.add_argument('--display_step', type=int, default="10", help='Check every number of update for disply the lossand early stopping (default: 10)')
    parser.add_argument('--stopping_criteria', type=int, default="15", help='exit training if validation error increase continously _ time (default: 5)')    
    

	# General parameters
    parser.add_argument('--seed', type=int, default="1", help="seed for rng (default: 1)")
    parser.add_argument("--device", type=str, default="/gpu:0", help="Compute device.")
    parser.add_argument('--allow-soft-placement', action='store_true', help='Place on GPU + CPU')
    parser.add_argument("--device-percentage", type=float, default="0.3", help="Amount of memory to use on device. (default: 0.3)")


	# actually parse the arguments
    return parser.parse_args()


## define the names for the log files that created during training
def get_name(model_name, update_num):
        '''return a unique name for this model,
           add more here to make more unique '''
        return "model_" + model_name + "_update_num_" + str(update_num) 
## read csv files 
def read_csv(filename):
    return genfromtxt(filename, delimiter=',')

## create directories
def create_dir(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)

## load the data
def load_data(data_dir):
    x = np.load(data_dir + 'x.npy')
    y = np.load(data_dir + 'y.npy')
    #S = np.load(data_dir + 'S.npy')
    noisy_S = np.load(data_dir + 'noisy_S.npy')
    np.random.seed(1)
    #latent_representaiton =  np.load(data_dir + 'latent_representation.npy')
    x_train_pre, x_test, y_train_pre, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train_pre,  y_train_pre, test_size=0.2)
    return x_train, x_val, x_test, y_train, y_val, y_test, noisy_S
	

## build the network 
def build_model(x, layer_sizes, is_training, output_dim, is_reuse):
    with tf.variable_scope("model"):
        winit = tf.contrib.layers.xavier_initializer()
        binit = tf.constant_initializer(0)
        activ_fn = tf.nn.sigmoid
        normalizer_fn = slim.batch_norm
        normalizer_params = {'is_training': is_training,
                             'decay': 0.999, 'center': True,
                             'scale': True, 'updates_collections': None}
        #keep_prob=0.5
        #is_training=is_training


        with slim.arg_scope([slim.fully_connected],
                                activation_fn=activ_fn,
                                weights_initializer=winit,
                                biases_initializer=binit,
                                weights_regularizer=slim.l2_regularizer(0.05),
                                reuse = is_reuse,
                                normalizer_fn=normalizer_fn,
                                normalizer_params=normalizer_params):
        
            # build the bulk of the layers
            layers = slim.stack(x, slim.fully_connected, layer_sizes, scope="layer")

            # final layer has NO activation, NO BN
            return slim.fully_connected(layers, output_dim,
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    weights_initializer=winit,
                                    biases_initializer=binit,
                                    weights_regularizer=slim.l2_regularizer(0.05),
                                    scope='prediction')

## augment the data 
def augment_data_y1_modified(batch_x, S):
    batch_x1 = np.copy(batch_x)
    batch_size = batch_x.shape[0]
    f1 = np.array([np.random.choice(r.nonzero()[0]) for r in batch_x])
    ss = S[f1,:]
    f2 = np.array([np.random.choice(r.nonzero()[0]) for r in ss]) # select one of the feature that is similar to the feature in f1
    row_index = np.arange(batch_size)
    batch_x1[row_index, f1] = 0
    batch_x1[row_index, f2] = batch_x[[row_index, f2]] + batch_x[row_index, f1]
    return batch_x1                 

 #3 augment by random pairs
def augment_Gaussian_noise(batch_x, sigma):
    batch_size = batch_x.shape[0]
    feature_dim = batch_x.shape[1]
    epsilon = np.random.normal(mu, sigma, [batch_size,feature_dim])
    epsilon[np.where(epsilon<0)] = 0
    batch_x1 = batch_x + epsilon
    return batch_x1           

## statistical test, need to instal:  pip install mlxtend
def Mcnemar_test(path1, path2, model1, model2):
    CDA_pred_y =np.load(path1 + '/pred_test.npy')
    DA_pred_y =np.load(path2 + '/pred_test.npy')
    true_y = np.load(path1 + '/test_y.npy')
    true_y = np.argmax(true_y, axis = 1)
    tb = mcnemar_table(y_target=true_y, 
                   y_model1=CDA_pred_y, 
                   y_model2=DA_pred_y)
    chi2, p = mcnemar(ary=tb, corrected=True)
    print('chi-squared:', chi2)
    print('p-value:', p)
    
    with open('/Mcnemar.csv', 'a') as newFile:
        newFileWriter = csv.writer(newFile)
        newFileWriter.writerow('Mecnemar test for' + model1 + 'and' + model2 + 'is' )
        newFileWriter.writerow(['pvalue:', p ])
          
'''
## functions might be needed later
## shuffle and split the data
def split_shuffle_data(X, Y, test_size):
    np.random.seed(1)
    x_train_pre, x_test, y_train_pre, y_test = train_test_split(X, Y, test_size=test_size)
    x_train, x_val, y_train, y_val = train_test_split(x_train_pre,  y_train_pre, test_size=test_size)
    return x_train, x_val, x_test, y_train, y_val, y_test

## augmented by random pairs
def augment_data_randompairs(batch_x):
    batch_size = batch_x.shape[0]
    feature_dim = batch_x.shape[1] 
    batch_x1= np.copy(batch_x)
    f1 = np.array([np.random.choice(r.nonzero()[0]) for r in batch_x]) 
    f2= np.random.randint(feature_dim, size=batch_size)
    row_index = np.arange(batch_size)
    batch_x1[row_index, f1] = batch_x[row_index, f2]
    batch_x1[row_index, f2] = batch_x[row_index, f1]
    return batch_x1



'''

