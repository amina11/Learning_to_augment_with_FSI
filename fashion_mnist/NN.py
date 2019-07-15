# Import necessary libraries
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
import os
import argparse
import csv


from util import get_fashion_MNIST_data, create_dir, build_model, NNModel, EarlyStopping, get_arguments


args = get_arguments()
# params that shoudl be specified
args.model_name = 'NN'
pathname = args.output_dir + args.model_name
args.reg_param1 = 0
args.reg_param2 = 0
args.epsilon = 0
create_dir(pathname)       

feature_dim = 784
output_dim = 10
#prepare the data
train_data, test_data = get_fashion_MNIST_data()
train_images, train_labels = train_data
train_images = train_images.reshape(train_images.shape[0], feature_dim)
train_labels = to_categorical(train_labels, num_classes=output_dim)
#train validation seperation
index = np.arange(train_images.shape[0])
np.random.shuffle(index)
train_x = train_images[index[:50000], :]
train_y = train_labels[index[:50000], :]
validation_x =train_images[index[50000:], :]
validation_y = train_labels[index[50000:], :]

test_images, test_labels = test_data
test_x = test_images.reshape(test_images.shape[0], feature_dim)
test_y = to_categorical(test_labels, num_classes=output_dim)

train_num = train_x.shape[0]
print('num_train', train_x.shape[0], 'num_validation', validation_x.shape[0], 'num_test', test_x.shape[0])


config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess = tf.Session(config = config)
# get the model
model = NNModel(sess, args)
# get the early stop
early_stop = EarlyStopping(model)
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)



update_num = 0
is_early_stop = False 
update_num = 0
args.batch_size = train_num
batch_num = int(train_num / args.batch_size)
model_name = 'base_model'
## base model training
for epoch in range(args.epochs):
    permutation = np.random.choice(train_num,train_num, replace=False)
    if is_early_stop == True:
        break
    for j in range(batch_num):
        update_num = update_num + 1
        batch_index = permutation[(j * args.batch_size): (j + 1) * args.batch_size]
        batch_x = train_x[batch_index, :]
        batch_y = train_y[batch_index, :]
        model.train_basemodel(sess, batch_x, batch_y, update_num, args.display_step)
        
        
        ##Early stopping starts after updates
        if update_num > args.start_early_stop and update_num % args.display_step == 0:
        	val_loss = model.val(sess, validation_x, validation_y)
        	is_early_stop = early_stop(sess, val_loss, args.stopping_criteria, pathname, model_name)
        	print(is_early_stop)
        	if is_early_stop:
        		early_stop.restore(sess, pathname, model_name)
        		break
model.test(sess, test_x, test_y, pathname, args.reg_param1, args.reg_param2, args.epsilon)

                
