# Import necessary libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time  # To time each epoch
from tensorflow.keras import datasets
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
import tensorflow.contrib.slim as slim
import os
import argparse
import csv

### define all the variables 
def get_arguments():

    parser = argparse.ArgumentParser(description='Three step optimization')
    #Model parameters
    parser.add_argument('--epsilon', type=float, help="epsilon to filter the augmented data")
    parser.add_argument('--reg_param1', type=float, help="regularizer parameter for y_y1")
    parser.add_argument('--reg_param2', type=float, help="regularizer parameter for fx1_y")
    parser.add_argument('--sigma', type=float, help="gaussian noise std")
    #input output paths
    #parser.add_argument('--data_name', type=str, help='name of the data we are calling')
    parser.add_argument('--data_dir', type=str, default='/work/amina/RawM_level_datacleaning2017_6_1/model/StabilityModel/Tensorflow_code/11_27_final/Test_on_text/ICDM/realworld_data/dataset/', help='directory which contains input data')
    parser.add_argument('--output_dir', type=str, default='./output/', help='directory for output of logs for tensorboard')
    parser.add_argument('--model_name', type=str, default='', help="the name of the model")
    #optimization parameters  
    parser.add_argument('--batch_size', type=int, default="500", help="batchsize (default: 50)")                  
    parser.add_argument('--epochs', type=int, default="700", help="optimization 1 epoch number (default: 36)")
    #parser.add_argument('--Save_every',  type=int, default="100", help='Save the train and validation loss at after __ update (default: 100)')
    parser.add_argument('--lr', type=float, default="1e-4", help='learning rate (default: 1e-3)')                    

    #early stopping parameters
    parser.add_argument('--start_early_stop', type=int, default="600", help='Activate early stopping after update_num (default:1900)')                           
    parser.add_argument('--display_step', type=int, default="10", help='Check every number of update for disply the lossand early stopping (default: 10)')
    parser.add_argument('--stopping_criteria', type=int, default="8", help='exit training if validation error increase continously _ time (default: 5)')    
    parser.add_argument('--seed', type=int, default="1", help="seed for rng (default: 1)")
    
    cmd_args, _ = parser.parse_known_args()
    return cmd_args


#1. prepare the data
def get_fashion_MNIST_data():
    """ Download Fashion MNIST dataset. """

    # Step 1: Get the Data
    train_data, test_data = datasets.fashion_mnist.load_data()  # Download FMNIST

    # Step 2: Preprocess Dataset
    """ Centering and Normalization
        Perform centering by mean subtraction, and normalization by dividing with 
        the standard deviation of the training dataset.
    """
    train_data_mean = np.mean(train_data[0])
    train_data_stdev = np.std(train_data[0])
    train_data = ((train_data[0] - train_data_mean) /
                  train_data_stdev, train_data[1])
    test_data = ((test_data[0] - train_data_mean) /
                 train_data_stdev, test_data[1])

    return train_data, test_data


def create_dir(pathname):
	if not os.path.exists(pathname):
		os.makedirs(pathname)

'''
#2. build the model
def build_model(x, output_dim):
    #h = Flatten(input_shape=(28, 28), name='flatten')(x)
    h = Dense(128, activation=tf.nn.relu, name='dense_1')(x)
    pred= Dense(output_dim, name='logits')(h)
    return pred

'''

## build the network 
def build_model(x, layer_sizes, is_training, output_dim, is_reuse):
    with tf.variable_scope("model"):
        winit = tf.contrib.layers.xavier_initializer()
        binit = tf.constant_initializer(0)
        activ_fn = tf.nn.relu
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
                                #weights_regularizer=slim.l2_regularizer(0.05),
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
                                    #weights_regularizer=slim.l2_regularizer(0.05),
                                    scope='prediction')

class NNModel(object):
    def __init__(self, sess, args):
        self.output_dim = 10
        self.feature_dim = 784
        #reg_param_1 = 1
        #reg_param_2 = 1
        #epsilon = 0.5
        #lr = 0.001
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, [None, self.feature_dim], name="input")
        self.y = tf.placeholder(tf.float32, [None, self.output_dim], name="output")
        self.y_output = build_model(self.x, [128], self.is_training, self.output_dim, None)
        #self.y_output =  build_model(self.x, output_dim)
        self.y_prob = tf.nn.softmax(self.y_output, name='prob')
        self.y_pred = tf.math.argmax(self.y_prob, 1, name='pred')
        self.correct_predictions = tf.equal(self.y_pred, tf.math.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
        self.loss_f =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_output, labels=self.y))
        self.optimizer_f = tf.train.AdamOptimizer(args.lr, name = "opt1").minimize(self.loss_f)

        self.x_1 = tf.placeholder(tf.float32, [None, self.feature_dim], name="input")
        #augmentation with filtering
        self.class_label = tf.placeholder (tf.float32, [None], name="condition_checking")
        self.y_1_output =  build_model(self.x_1, [128], self.is_training, self.output_dim, True)
        self.cond =  tf.reduce_sum(tf.squared_difference(self.y_1_output, self.y_output), 1)
        self.row_index = tf.where(self.class_label > 0)
        self.y1_filterred = tf.squeeze(tf.gather(self.y_1_output, self.row_index))
        self.y_outout_filtered = tf.squeeze(tf.gather(self.y_output, self.row_index))
        self.y_filtered =  tf.squeeze(tf.gather(self.y, self.row_index))
        self.is_empty = tf.equal(tf.size(self.row_index), 0)

        self.loss_y_y1_filtered = self.output_dim * tf.reduce_mean(tf.squared_difference(self.y_outout_filtered, self.y1_filterred), name="loss_f_G_filtered")
        self.loss_y_y1_filtered = tf.cond(tf.cast(self.is_empty, tf.bool), lambda: tf.constant(0, tf.float32), lambda:self.loss_y_y1_filtered) #then corresponding loss is zero, in this way avoid nan
        
        self.loss_fx_y= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y1_filterred, labels=self.y_filtered), name = "filtered_reg")
        self.loss_fx_y_filtered = tf.cond(tf.cast(self.is_empty, tf.bool), lambda: tf.constant(0, tf.float32), lambda:self.loss_fx_y) #then corresponding loss is zero, in this way avoid nan
        
        self.final_reg_filtered = tf.add(args.reg_param1 * self.loss_y_y1_filtered, args.reg_param2 * self.loss_fx_y_filtered, name="filtered_loss")
        self.loss_final_filtered = tf.add(self.final_reg_filtered, self.loss_f, name="final_filteded")
        self.optimizer_final_filtered = tf.train.AdamOptimizer(args.lr, name = "opt2").minimize(self.loss_final_filtered) 


        ## augmentation without filtering
        self.loss_y_y1 = self.output_dim * tf.reduce_mean(tf.math.squared_difference(self.y_1_output, self.y), name="loss_f_G")
        self.loss_augment =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_1_output, labels=self.y))
        self.final_reg = tf.add(args.reg_param1 * self.loss_y_y1, args.reg_param2 * self.loss_augment, name="reg_final")

        self.loss_final = tf.add(self.final_reg, self.loss_f, name="final_loss")
        self.optimizer_final = tf.train.AdamOptimizer(args.lr, name = "opt2").minimize(self.loss_final) 
        self.saver = tf.train.Saver()

    def restore(self, sess, output_dir, model_name):
        model_filename = "{}/models/{}.ckpt".format(output_dir,model_name)
        print('restoring model from %s...' % model_filename)
        self.saver.restore(sess, model_filename)         

    def save(self, sess, output_dir,model_name, overwrite=True):
        model_filename = "{}/models/{}.ckpt".format(output_dir, model_name)
        if not os.path.isfile(model_filename) or overwrite:
            print('saving model to %s...' % model_filename)
            self.saver.save(sess, model_filename)


    # train for the base model 
    def train_basemodel(self, sess, batch_x, batch_y, update_num, display_step=1):
        feed_dict={self.x: batch_x, self.y: batch_y, self.is_training: True}
        if update_num % display_step ==0:
        	loss, _ = sess.run([self.loss_f, self.optimizer_f], feed_dict=feed_dict)
        	print("[update_number:", '%04d]' % (update_num), "training loss = ", "{:.4f} | ".format(loss))
        else:
        	_ = sess.run(self.optimizer_f, feed_dict=feed_dict)



    def standard_augmentation(self, sess, batch_x, batch_x1, batch_y, S , update_num, display_step=1):
        batch_x1= augment_data_y1_modified(batch_x, S)
        if update_num % display_step ==0:
        	feed_dict={self.x: batch_x, self.y: batch_y, self.x_1: batch_x1, self.is_training: True}
        	loss, _ = sess.run([self.loss_final, self.optimizer_final], feed_dict=feed_dict)
        	print("[update_number:", '%04d]' % (update_num),"training loss = ", "{:.4f} | ".format(loss))
        else: 
            _ = sess.run(self.optimizer_final, feed_dict=feed_dict)

    def gaussian_augmentation(self, sess, batch_x, batch_y, args, update_num):
        batch_x1= augment_Gaussian_noise(batch_x, args.sigma)
        feed_dict={self.x: batch_x, self.y: batch_y, self.x_1: batch_x1,self.is_training: True}
        if update_num % args.display_step == 0:
            loss,  _ = sess.run([self.loss_final, self.optimizer_final], feed_dict=feed_dict)
            print("[update_number:", '%04d]' % (update_num),"training loss = ", "{:.4f} | ".format(loss))
        else:
            _ = sess.run(self.optimizer_final, feed_dict=feed_dict)
            

        

    def gaussian_augmentation_filtering(self, sess, batch_x, batch_y, args,update_num):
        batch_x1= augment_Gaussian_noise(batch_x, args.sigma)
        class_label1 = np.zeros([args.batch_size], dtype = 'float32')
        cond1 = sess.run(self.cond, feed_dict={self.x_1: batch_x1, self.x: batch_x, self.is_training: False})
        class_label1[[np.where(cond1 <= args.epsilon)]] = 1
        class_label1[[np.where(cond1 > args.epsilon)]] = 0
        feed_dict={self.x: batch_x, self.y: batch_y, self.x_1: batch_x1,  self.class_label: class_label1, self.is_training: True}
        ## Display the logs every kth update
        if update_num % args.display_step == 0:
            loss, _ = sess.run([self.loss_final_filtered, self.optimizer_final_filtered], feed_dict=feed_dict)
            print("[update_number:", '%04d]' % (update_num),"training loss = ", "{:.4f} | ".format(loss))
        else:
            _ = sess.run(self.optimizer_final_filtered, feed_dict=feed_dict) 
        

    def augment_similar_pairs(self,sess, batch_x, batch_y, args, update_num, S, pairs_num):
        batch_x1= augment_multiple_pairs(batch_x, S, pairs_num)
        class_label1 = np.zeros([args.batch_size], dtype = 'float32')
        cond1 = sess.run(self.cond, feed_dict={self.x_1: batch_x1, self.x: batch_x, self.is_training: False})
        class_label1[[np.where(cond1 <= args.epsilon)]] = 1
        class_label1[[np.where(cond1 > args.epsilon)]] = 0
        feed_dict={self.x: batch_x, self.y: batch_y, self.x_1: batch_x1,  self.class_label: class_label1, self.is_training: True}
        ## Display the logs every kth update
        if update_num % args.display_step == 0:
            loss, _ = sess.run([self.loss_final_filtered, self.optimizer_final_filtered], feed_dict=feed_dict)
            print("[update_number:", '%04d]' % (update_num),"training loss = ", "{:.4f} | ".format(loss))
        else:
            _ = sess.run(self.optimizer_final_filtered, feed_dict=feed_dict) 




    # evalute on the validation set, keep the summaries
    def val(self, sess, val_x, val_y):
        #loss_vector = []
        feed_dict = {self.x: val_x, self.y: val_y, self.is_training: False}
        # get all metrics here
        loss  = sess.run(self.loss_f, feed_dict=feed_dict)
        #loss_vector.append(loss)
        print("validation loss = ", "{:.4f} | ".format(loss))
        return loss

    # test and save the best result   
    def test(self, sess, test_x, test_y, output_dir,reg_param1, reg_param2, epsilon):
        feed_dict = {self.x: test_x, self.y: test_y,self.is_training: False}
        loss, accuracy, pred_test_y  = sess.run([self.loss_f, self.accuracy, self.y_pred], feed_dict=feed_dict)
        print("test loss is =", "{:.4f} | ".format(loss), "test accuracy is =", "{:.4f} | ".format(accuracy))
        np.save(output_dir + '/pred_test_y.npy', pred_test_y)
        np.save(output_dir + '/test_y', test_y)

        with open(output_dir + '/Final_result.csv', 'a') as newFile:
            headers = ['epsilon', 'reg_param1', 'reg_param2', 'test_loss', 'test_accuracy']
            newFileWriter = csv.DictWriter(newFile, fieldnames=headers)
            newFileWriter.writeheader()
            newFileWriter.writerow({'epsilon': epsilon, 'reg_param1': reg_param1, 'reg_param2': reg_param2, 'test_loss': loss, 'test_accuracy': accuracy})


## 6.define early stopping         
class EarlyStopping(object):
    def __init__(self, model,save_best=True):
        self.model = model
        self.save_best = save_best
        self.loss = 0.0
        self.iteration = 0
        self.stopping_step = 0
        self.best_loss = np.inf
        
    
    def restore(self, sess, output_dir,model_name):
        self.model.restore( sess, output_dir, model_name)

    def __call__(self, sess, loss, stopping_criteria, output_dir, model_name):
        is_early_stop = False
        print(self.stopping_step)
        if (loss < self.best_loss):
            self.stopping_step = 0
            self.best_loss = loss
            if self.save_best:
                self.model.save(sess, output_dir,model_name)
        else:
            self.stopping_step += 1

            
        if self.stopping_step >= stopping_criteria:
            print("Early stopping is triggered;  loss:{} | iter: {}".format(loss, self.iteration))
            print(self.stopping_step)
            is_early_stop = True

        self.iteration += 1
        return is_early_stop      
        

#augment by random pairs
def augment_Gaussian_noise(batch_x, sigma):
    batch_size = batch_x.shape[0]
    feature_dim = batch_x.shape[1]
    epsilon = np.random.normal(0, sigma, [batch_size,feature_dim])
    batch_x1 = batch_x + epsilon
    return batch_x1           

#augment by similar pairs
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


def augment_multiple_pairs(batch_x, S, pairs_num):
    batch_x1 = np.copy(batch_x)
    batch_size = batch_x.shape[0]
    f1 = np.array(np.random.choice(784, pairs_num))
    f2 = S[f1,:]
    f2 = np.array([np.random.choice(r.nonzero()[0]) for r in f2])
    batch_x1[:, f1] = 0
    batch_x1[:, f2] = batch_x[:, f2] + batch_x[:, f1]
    return batch_x1




if __name__ == '__main__':
    pass
