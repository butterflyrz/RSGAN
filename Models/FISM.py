from __future__ import absolute_import
from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

class FISM(object):
    def __init__(self, num_users, num_items, args):
        self.loss_func = args.loss_func
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.eval = args.eval

        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.alpha = args.alpha

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None])	#the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None, 1])	#the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1])	  #the index of items
            self.labels = tf.placeholder(tf.float32, shape=[None,1])	#the ground truth

    def _create_variables(self):
        with tf.name_scope("embedding"):  # The embedding initialization is unknown now
            self.c1 = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                 name='c1', dtype=tf.float32)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2' )
            self.embedding_Q_ = tf.concat([self.c1,self.c2], 0, name='embedding_Q_')
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_Q', dtype=tf.float32)
            self.bias = tf.Variable(tf.zeros(self.num_items),name='bias')

    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q_, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)
            self.bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
            self.coeff = tf.pow(self.num_idx, -tf.constant(self.alpha, tf.float32, [1]))
            self.output = tf.sigmoid(self.coeff * tf.expand_dims(tf.reduce_sum(self.embedding_p*self.embedding_q, 1),1) + self.bias_i)

    def _create_loss(self):
        with tf.name_scope("loss"):
            if self.loss_func == "logloss":
                self.loss = tf.losses.log_loss(self.labels, self.output) + \
                            self.lambda_bilinear*tf.reduce_sum(tf.square(self.embedding_Q)) + self.gamma_bilinear*tf.reduce_sum(tf.square(self.embedding_Q_))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            if self.loss_func == "logloss":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            else :
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()