
### this file is made for critical data augmentation model y - f(x1) 
## condition f(x) - f(x1)
## auther amina

from __future__ import division
import os
import csv
import scipy.io
import numpy as np
import tensorflow as tf
import utilities as am_util
import tensorflow.contrib.slim as slim

## 1. get the arguments
args = am_util.get_arguments()
np.random.seed(args.seed)
## 2. load the data
train_xx, train_yy, val_x, val_y, test_x, test_y, S_30 = am_util.load_data(args.data_dir + args.data_name + '/')
zero_row = np.where(~train_xx.any(axis=1))[0]
train_x = np.delete(train_xx, (zero_row), axis=0)
train_y = np.delete(train_yy,(zero_row ), axis=0)

[train_num, feature_dim] = train_x.shape
[train_num, output_dim] = train_y.shape
batch_num = int(train_num / args.batch_size)

## 3. fix seed for graph and numpy operations
tf.reset_default_graph()
tf.set_random_seed(args.seed)

## 4.create output folders to save the log files and trained model
#main_output_dir = ./output/data_name/
am_util.create_dir(os.path.join(args.output_dir + args.data_name)) # create forlder for output
main_output_dir = args.output_dir + args.data_name
##./output/data_name/model_name/
modelname = args.model_name +'_epsilon_' + str(args.epsilon) + '_reg_param1_' + str(args.reg_param1) +  '_reg_param2_' + str(args.reg_param2)
am_util.create_dir(os.path.join(args.output_dir, args.data_name, modelname)) ## to save predition result and log files
output_dir = main_output_dir + '/modelname/'
##./output/data_name/model_name/models
output_model_dir = args.output_dir + args.data_name + '/'+  modelname + "/models" ## forlder to save the model under ourput folder
am_util.create_dir(os.path.join(args.output_dir, args.data_name, modelname, "models"))  
##./output/data_name/model_name/logs
output_log_dir = args.output_dir + args.data_name + '/'+  modelname + "/logs" ## forlder to save the model under ourput folder
am_util.create_dir(os.path.join(args.output_dir, args.data_name, modelname, "logs"))  


## 5.define the full graph
class NNModel(object):
    def __init__(self, sess, args):
        ''' the main neural network model class '''
        #self.config = vars(args)
        self.x = tf.placeholder(tf.float32, [None, feature_dim], name="input")
        self.y_ = tf.placeholder(tf.float32, [None, output_dim], name="output")
        self.is_training = tf.placeholder(tf.bool)
        ## for the augmented data
        self.x1 = tf.placeholder(tf.float32, [None, feature_dim], name="input")
        self.class_label = tf.placeholder (tf.float32, [None], name="condition_checking")
        self.layer_sizes = [args.net1_h1, args.net1_h2]
        ## build the model
        self.y = am_util.build_model(self.x, self.layer_sizes, self.is_training, output_dim, None)  # reuse none so that the variables are created
        self.prob = tf.nn.softmax(self.y, name='prob')
        self.pred = tf.arg_max(self.prob, 1, name='pred')
        
        ##accuarcy 
        self.correct_predictions = tf.equal(self.pred, tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
        #loss and optimizer
        self.loss_f =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
        self.optimizer_f = tf.train.AdamOptimizer(args.lr, name = "opt1").minimize(self.loss_f)
        
        ## build all the summaries and writers
        self.summaries = tf.summary.merge(self.get_summaries())
        self.train_summary_writer = tf.summary.FileWriter("%s/logs/train" % output_dir,
                                                          sess.graph,
                                                          flush_secs=60)
        self.val_summary_writer = tf.summary.FileWriter("%s/logs/val" % output_dir,
                                                        sess.graph,
                                                        flush_secs=60)
        self.test_summary_writer = tf.summary.FileWriter("%s/logs/test" % output_dir,
                                                         sess.graph,
                                                         flush_secs=60)
        # build a saver object
        self.saver = tf.train.Saver(tf.global_variables() + tf.local_variables())
        
        # for augmented data 
        self.y1 = am_util.build_model(self.x1, self.layer_sizes, self.is_training, output_dim, True) # reuse true so that the variables are shared from previosly builded network
        self.cond =  tf.reduce_sum(tf.squared_difference(self.y1, self.y), 1)
        self.row_index = tf.where(self.class_label > 0)
        self.y1_filterred = tf.squeeze(tf.gather(self.y1, self.row_index))
        self.y_filtered = tf.squeeze(tf.gather(self.y, self.row_index))
        self.is_empty = tf.equal(tf.size(self.row_index), 0)
        self.loss_y_y1 = output_dim * tf.reduce_mean(tf.squared_difference(self.y_filtered, self.y1_filterred), name="loss_f_G")
        self.loss_y_y1_filtered = tf.cond(tf.cast(self.is_empty, tf.bool), lambda: tf.constant(0, tf.float32), lambda:self.loss_y_y1) #then corresponding loss is zero, in this way avoid nan
        self.y__filtered = tf.squeeze(tf.gather(self.y_, self.row_index))
        self.loss_fx_y= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y1_filterred, labels=self.y__filtered), name = "filtered_reg")
        self.loss_fx_y_filtered = tf.cond(tf.cast(self.is_empty, tf.bool), lambda: tf.constant(0, tf.float32), lambda:self.loss_fx_y) #then corresponding loss is zero, in this way avoid nan
        
        #final loss
        self.final_reg = tf.add(args.reg_param1 * self.loss_y_y1_filtered, args.reg_param2 * self.loss_fx_y_filtered, name="loss_final")
        self.loss_final = tf.add(self.final_reg, self.loss_f, name="loss_final")
        self.optimizer_final = tf.train.AdamOptimizer(args.lr, name = "opt2").minimize(self.loss_final) 
        
    # get  summaries for the tensorboard   
    def get_summaries(self):
        return [
            tf.summary.scalar("loss", self.loss_f)]
    
    # save the model
    def save(self, sess, model_name, update_num, overwrite=False):
        model_filename = "{}/models/{}.cpkt".format(
            output_model_dir, am_util.get_name(model_name, update_num)
        )
        if not os.path.isfile(model_filename) or overwrite:
            print('saving model to %s...' % model_filename)
            self.saver.save(sess, model_filename)
            
            
    # restore the model        
    def restore(self, sess, model_name, update_num):
        model_filename = "{}/models/{}.cpkt".format(
            output_model_dir, am_util.get_name(model_name, update_num)
        )
        if os.path.isfile(model_filename):
            print('restoring model from %s...' % model_filename)
            self.saver.restore(sess, model_filename)         
    
    # train for the base model 
    def train_basemodel(self, sess, batch_x, batch_y, update_num, grapher,  display_step=10):
        feed_dict={self.x: batch_x, self.y_: batch_y, self.is_training: True}
        
        ## Display the logs every kth update
        if update_num % display_step == 0:
            loss, summaries, _ = sess.run([self.loss_f, self.summaries, self.optimizer_f], feed_dict=feed_dict)
            print("[update_number:", '%04d]' % (update_num),"training loss = ", "{:.4f} | ".format(loss))
            grapher.add_summary(summaries, update_num)
        else:
            _ = sess.run(self.optimizer_f, feed_dict=feed_dict)
            
    # final train with the augmented data        
    def train_final(self, sess, batch_x, batch_y, update_num, grapher, epsilon, display_step=10):
        batch_x1= am_util.augment_data_y1_modified(batch_x, S_30)
        class_label1 = np.zeros([args.batch_size], dtype = 'float32')
        cond1 = sess.run(self.cond, feed_dict={self.x1: batch_x1, self.x: batch_x, self.is_training: False})
        class_label1[[np.where(cond1 <= epsilon)]] = 1
        class_label1[[np.where(cond1 > epsilon)]] = 0
        feed_dict={self.x: batch_x, self.y_: batch_y, self.x1: batch_x1,  self.class_label: class_label1, self.is_training: True}
        ## Display the logs every kth update
        if update_num % display_step == 0:
            loss, summaries, _ = sess.run([self.loss_final, self.summaries, self.optimizer_final], feed_dict=feed_dict)
            print("[update_number:", '%04d]' % (update_num),"training loss = ", "{:.4f} | ".format(loss))
            grapher.add_summary(summaries, update_num)
        else:
            _ = sess.run(self.optimizer_final, feed_dict=feed_dict)
            
            
        
    # evalute on the validation set, keep the summaries
    def val(self, sess, val_x, val_y, update_num, grapher, display_step=10):
        #loss_vector = []
        feed_dict = {self.x: val_x, self.y_: val_y, self.is_training: False}
        # get all metrics here
        if update_num % display_step == 0:
            loss  = sess.run(self.loss_f, feed_dict=feed_dict)
            #loss_vector.append(loss)
            print("[update_number:", '%04d]' % (update_num),"validation loss = ", "{:.4f} | ".format(loss))
            loss_summary = tf.Summary()
            loss_summary.value.add(tag="test_or_val_loss", simple_value=loss)
            grapher.add_summary(loss_summary, update_num)
            return loss
    
    # test and save the best result   
    def test(self, sess, test_x, test_y, output_dir):
        feed_dict = {self.x: test_x, self.y_: test_y, self.is_training: False}
        loss, accuracy, pred_test_y  = sess.run([self.loss_f, self.accuracy, self.pred], feed_dict=feed_dict)
        print("test loss is =", "{:.4f} | ".format(loss), "test accuracy is =", "{:.4f} | ".format(accuracy))
        np.save(output_dir + '/pred_test_y.npy', pred_test_y)
        np.save(output_dir + '/test_y', test_y)
        
        with open(output_dir + '/Final_result.csv', 'a') as newFile:
            headers = ['epsilon', 'reg_param1', 'reg_param2', 'test_loss', 'test_accuracy']
            newFileWriter = csv.DictWriter(newFile, fieldnames=headers)
            newFileWriter.writeheader()
            newFileWriter.writerow({'epsilon': args.epsilon, 'reg_param1': args.reg_param1, 'reg_param2': args.reg_param2, 'test_loss': loss, 'test_accuracy': accuracy})
    
        
## 6.define early stopping         
class EarlyStopping(object):
    def __init__(self, model, save_best=True):
        self.model = model
        self.save_best = save_best
        self.loss = 0.0
        self.iteration = 0
        self.stopping_step = 0
        self.best_loss = np.inf
    
    def restore(self, sess, model_name, update_num):
        self.model.restore( sess, model_name, update_num)

    def __call__(self, loss, stopping_criteria, model_name, update_num):
        is_early_stop = False
        print(self.stopping_step)
        if (loss < self.best_loss):
            self.stopping_step = 0
            self.best_loss = loss
            if self.save_best:
                self.model.save(sess, model_name, update_num, overwrite=True)
        else:
            self.stopping_step += 1

            
        if self.stopping_step >= stopping_criteria:
            print("Early stopping is triggered;  loss:{} | iter: {}".format(loss, self.iteration))
            print(self.stopping_step)
            is_early_stop = True

        self.iteration += 1
        return is_early_stop      
        
    

## 7. define the session
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess = tf.Session(config = config)
# get the model
model = NNModel(sess, args)
# get the early stop
early_stop = EarlyStopping(model)
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)


## 8. start training for the base model
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


## 9. start training for the final model with the augmented data 
is_early_stop = False
update_num = 0
for epoch in range(args.epochs):
    permutation = np.random.choice(train_num,train_num, replace=False)
    if is_early_stop == True:
            break
    for j in range(batch_num):
        update_num = update_num + 1
        batch_index = permutation[(j * args.batch_size): (j + 1) * args.batch_size]
        batch_x = train_x[batch_index, :]
        batch_y = train_y[batch_index, :]
        
        model.train_final(sess, batch_x, batch_y, update_num,  model.train_summary_writer, args.epsilon, args.display_step)
        val_loss = model.val(sess, val_x, val_y, update_num, model.val_summary_writer, args.display_step)
        
        ##Early stopping 
        if update_num > args.start_early_stop and update_num % args.display_step == 0:
            is_early_stop = early_stop(val_loss, args.stopping_criteria, args.model_name, update_num)
            print(is_early_stop)
            if is_early_stop:
                early_stop.restore(sess, args.model_name, update_num)
                model.test(sess, test_x, test_y, output_dir)
                break
 

