from __future__ import absolute_import
from __future__ import division

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


class GMF:
    def __init__(self, num_items, num_users, args):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embedding_size

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = "user_input")
            self.item_input = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input")
            self.labels = tf.placeholder(tf.float32, shape = [None, 1], name = "labels")
    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_P', dtype=tf.float32)
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_Q', dtype=tf.float32)
            self.h = tf.Variable(tf.zeros(self.embedding_size),name='h')  //how to initialize it
    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)
            self.output = tf.sigmoid(tf.matmul(self.h, self.embedding_p*self.embedding_q)

    def _create_loss(self):
        with tf.name_scope("loss"):
            if loss_func == "logloss":
                self.loss = tf.losses.log_loss(self.labels, self.output) + self.lambda_bilinear * tf.reduce_sum(
                                tf.square(self.embedding_Q)) + self.gamma_bilinear * tf.reduce_sum(tf.square(self.embedding_Q_))
            elif loss_func == "BPR":
            else :
                print "Don't build Loss Function!"

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()

class MLP:
    def __init__(self, num_items, num_users, args):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embedding_size
        self.num_layer = args.num_layer
        self.weight_size = args.weight_size

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape = [None, 1], name = "user_input")
            self.item_input = tf.placeholder(tf.int32, shape = [None, 1], name = "item_input")
            self.labels = tf.placeholder(tf.float32, shape = [None, 1], name = "labels")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_P', dtype=tf.float32)  #(num_users, embedding_size)
            self.embedding_Q = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                                                name='embedding_Q', dtype=tf.float32)
            self.h = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size[-1]], mean=0.0, stddev=0.01),
                                 dtype=tf.float32, name='h')  #(1,w[-1]) how to initialize it
            self.W, self.b = [], []
            self.weight_sizes = [self.num_users + self.num_items] + self.weight_size
            for i in range(self.num_layer):
                W_temp = tf.Variable(tf.truncated_normal(shape=[self.weight_size[i+1], self.weight_size[i]], mean=0.0, stddev=0.01),
                                                                name='W' + str(i), dtype=tf.float32) #(w[1], 2*embed_size) (w[i+1],w[i])
                b_temp = tf.Variable(tf.zeros(self.weight_size[i+1]), dtype=tf.float32, name='b' + str(i))  # (w[i], )
                self.W.append(W_temp)
                self.b.append(b_temp)

    def _create_inference(self):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)  #(embedding_size,)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)  #(embedding_size,)

            self.z = []
            z_temp = tf.expand_dims(tf.concat([self.embedding_p, self.embedding_q]),1)  #(2*embed_size, 1)
            self.z.append(z_temp)

            for i in range(self.num_layer):
                z_temp = tf.nn.relu(tf.matmul(self.W[i], self.z[i]) + self.b[i]) #(W[i+1], 1)
                self.z.append(z_temp)

            self.output = tf.sigmoid(tf.matmul(self.h, self.z[-1])) # 1

    def _create_loss(self):
        with tf.name_scope("loss"):
            if self.loss_func == "logloss":
                self.loss = tf.losses.log_loss(self.labels, self.output) + \
                            self.lambda_bilinear*tf.reduce_sum(tf.square(self.embedding_Q)) + self.gamma_bilinear*tf.reduce_sum(tf.square(self.embedding_Q_))
            elif self.loss_func == "BPR":
                self.loss =
            else:
                print "Don't build loss function!"

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()

class FISM:
    def __init__(self, num_items, args):
        self.num_items = num_items
        self.dataset_name = args.dataset
        self.learning_rate = args.lr
        self.embedding_size = args.embed_size
        self.alpha = args.alpha
        self.verbose = args.verbose
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.batch_choice = args.batch_choice
        self.train_loss = args.train_loss
        self.logAttention = 0
        self.ckpt = args.checkpoint
        self.loss_func = args.loss

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
            elif self.loss_func == "BPR":
                self.loss =
            else:
                print "Don't build loss function!"

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()