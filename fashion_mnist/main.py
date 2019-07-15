# Import necessary libraries
import numpy as np
import sonnet as snt
import tensorflow as tf
import matplotlib.pyplot as plt
import time  # To time each epoch
from tensorflow.keras import datasets
print("Tensorflow version: {}".format(tf.__version__))
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
import os

from util import get_fashion_MNIST_data, create_dir
 #prepare the data
feature_dim = 784
output_dim = 10
output_dir = './output'
pathname = output_dir + 'models'
create_dir(pathname)       



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


#2. build the model
def build_model(x, output_dim):
  #h = Flatten(input_shape=(28, 28), name='flatten')(x)
  h = Dense(128, activation=tf.nn.relu, name='dense_1')(x)
  pred= Dense(output_dim, name='logits')(h)
  return pred
 


class NNModel(object):
    def __init__(self, sess, args):
 
		self.x = tf.placeholder(tf.float32, [None, args.feature_dim], name="input")
		self.y = tf.placeholder(tf.float32, [None, args.output_dim], name="output")
		self.y_output =  build_model(self.x, args.output_dim)
		self.y_prob = tf.nn.softmax(self.y_output, name='prob')
		self.y_pred = tf.arg_max(self.y_prob, 1, name='pred')
		self.correct_predictions = tf.equal(self.y_pred, tf.math.argmax(self.y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
		self.loss_f =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_output, labels=self.y))

		self.x_1 = tf.placeholder(tf.float32, [None, args.feature_dim], name="input")
		self.y_1_output =  build_model(self.x_1, args.output_dim)
		self.loss_augment =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_1_output, labels=self.y))
		self.final_reg = tf.add(args.reg_param_1 * self.loss_y_y1, args.reg_param_2 * self.loss_augment, name="loss_final")

		self.loss_final = tf.add(self.final_reg, self.loss_f, name="loss_final")
		self.optimizer_final = tf.train.AdamOptimizer(args.lr, name = "opt2").minimize(loss_final) 
        

	 def restore(self, sess, output_dir, model_name):
      model_filename = "{}/models/{}.cpkt".format(output_dir,model_name)
      if os.path.isfile(model_filename):
            print('restoring model from %s...' % model_filename)
            self.saver.restore(sess, model_filename)         
    
    def save(self, sess, output_dir,model_name,overwrite=True):
        model_filename = "{}/models/{}.cpkt".format(output_dir,model_name)
        if not os.path.isfile(model_filename) or overwrite:
            print('saving model to %s...' % model_filename)
            self.saver.save(sess, model_filename)
	

	# train for the base model 
    def train_basemodel(self, sess, batch_x, batch_y, update_num):
        feed_dict={self.x: batch_x, self.y: batch_y}
        _ = sess.run(self.optimizer_f, feed_dict=feed_dict)


	def standard_augmentation(self, sess, batch_x, batch_x1, batch_y):
        batch_x1= am_util.augment_data_y1_modified(batch_x, S_30)
        feed_dict={self.x: batch_x, self.y: batch_y, self.x_1: batch_x1}
        _ = sess.run(self.optimizer_final, feed_dict=feed_dict)
            
       
    # evalute on the validation set, keep the summaries
    def val(self, sess, val_x, val_y):
        #loss_vector = []
        feed_dict = {self.x: val_x, self.y_: val_y}
        # get all metrics here
       
        loss  = sess.run(self.loss_f, feed_dict=feed_dict)
        #loss_vector.append(loss)
        print("validation loss = ", "{:.4f} | ".format(loss))
        return loss
    
    

    # test and save the best result   
    def test(self, sess, test_x, test_y, output_dir):
        feed_dict = {self.x: test_x, self.y_: test_y}
        loss, accuracy, pred_test_y  = sess.run([self.loss_f, self.accuracy, self.pred], feed_dict=feed_dict)
        print("test loss is =", "{:.4f} | ".format(loss), "test accuracy is =", "{:.4f} | ".format(accuracy))
        np.save(output_dir + '/pred_test_y.npy', pred_test_y)
        np.save(output_dir + '/test_y', test_y)
        
        with open(output_dir + '/Final_result.csv', 'a') as newFile:
            headers = ['epsilon', 'reg_param1', 'reg_param2', 'test_loss', 'test_accuracy']
            newFileWriter = csv.DictWriter(newFile, fieldnames=headers)
            newFileWriter.writeheader()
            newFileWriter.writerow({'epsilon': args.epsilon, 'reg_param1': args.reg_param1, 'reg_param2': args.reg_param2, 'test_loss': loss, 'test_accuracy': accuracy})
    
## 7. define the session
from util import EarlyStopping
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
        model.train_basemodel(sess, batch_x, batch_y, update_num,  model.train_summary_writer, args.display_step)
        val_loss = model.val(sess, val_x, val_y, update_num, model.val_summary_writer, args.display_step)
        
        ##Early stopping starts after updates
        if update_num > args.start_early_stop and update_num % args.display_step == 0:
            is_early_stop = early_stop(val_loss, args.stopping_criteria, args.model_name, update_num)
            print(is_early_stop)
            if is_early_stop:
                early_stop.restore(sess, args.model_name, update_num)
                #model.test(sess, test_x, test_y, output_dir)
                break
