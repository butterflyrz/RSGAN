from __future__ import absolute_import
from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

class MLP(object):
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
        self.weight_size = eval(args.layer_size)
        self.num_layer = len(self.weight_size)

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = "user_input")  #(b, 1)
            self.item_input = tf.placeholder(tf.int32, shape = [None, None], name = "item_input")
            self.labels = tf.placeholder(tf.float32, shape = [None, 1], name = "labels")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_P', dtype=tf.float32)  #(num_users, embedding_size)
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_Q', dtype=tf.float32)  #(num_items, embedding_size)
            self.h = tf.Variable(tf.random_uniform([self.weight_size[-1], 1], minval = -tf.sqrt(3/self.weight_size[-1]),
                                                   maxval = tf.sqrt(3/self.weight_size[-1])), name = 'h') #(W[-1], 1)
            self.W, self.b = {}, {}
            self.weight_sizes = [2 * self.embedding_size] + self.weight_size
            for i in range(self.num_layer):
                self.W[i] = tf.Variable(tf.random_uniform(shape=[self.weight_sizes[i], self.weight_sizes[i+1]], minval = -tf.sqrt(6/(self.weight_sizes[i]+self.weight_sizes[i+1])),
                                                   maxval = tf.sqrt(6/(self.weight_sizes[i]+self.weight_sizes[i+1]))), name='W' + str(i), dtype=tf.float32) #(2*embed_size, W[1]) (w[i],w[i+1])
                self.b[i] = tf.Variable(tf.zeros([1,self.weight_sizes[i+1]]), dtype=tf.float32, name='b' + str(i))  # (1, W[i+1])

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input),1)  #(b, embedding_size)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input),1)  #(b, embedding_size)

            self.z = []
            z_temp = tf.concat([self.embedding_p, self.embedding_q], 1)  #(b, 2*embed_size)
            self.z.append(z_temp)

            for i in range(self.num_layer):
                z_temp = tf.nn.relu(tf.matmul(self.z[i], self.W[i]) + self.b[i]) #(b, W[i]) * (W[i], W[i+1]) + (1, W[i+1]) => (b, W[i+1])
                self.z.append(z_temp)
            return tf.sigmoid(tf.matmul(z_temp, self.h)) # (b, W[-1]) * (W[-1], 1) => (b, 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            if self.loss_func == "logloss":
                self.output = self._create_inference(self.item_input)
                self.loss = tf.losses.log_loss(self.labels, self.output) + \
                            self.lambda_bilinear * tf.reduce_sum(tf.square(self.embedding_P)) + \
                            self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q))
            else:
                self.output = self._create_inference(tf.expand_dims(self.item_input[:, 0], axis = 1))
                self.output_neg = self._create_inference(tf.expand_dims(self.item_input[:, -1], axis = 1))
                self.result = self.output - self.output_neg
                self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) \
                                    + self.lambda_bilinear*tf.reduce_sum(tf.square(self.embedding_P)) \
                                    + self.gamma_bilinear*tf.reduce_sum(tf.square(self.embedding_Q))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            if self.loss_func == "logloss":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            else :
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
