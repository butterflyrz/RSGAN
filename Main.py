import os

import tensorflow as tf
import numpy as np

import logging

from time import time
from time import strftime
from time import localtime

from Models import FISM
from Models import GMF
from Models import MLP

import BatchGen.BatchGenItem as BatchItem
import BatchGen.BatchGenUser as BatchUser

import Evaluate.EvaluateItem as EvalItem
import Evaluate.EvaluateUser as EvalUser

from Dataset import Dataset

import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args():
    parser = argparse.ArgumentParser(description="Run RSGAN.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='yelp',
                        help='Choose a dataset.')
    parser.add_argument('--model', nargs='?', default='GMF',
                        help='Choose model: GMF, MLP, FISM')
    parser.add_argument('--loss_func', nargs='?', default='logloss',
                        help='Choose loss: logloss, BPR')
    parser.add_argument('--batch_gen', nargs='?', default='unfixed',
                        help='Coose batch gen: fixed, unfixed')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', type=int, default='256',
                        help='batch_size')
    parser.add_argument('--batch_choice', nargs='?', default='user',
                        help='user: generate batches by user, fixed:batch_size: generate batches by batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[32,16,8]',
                        help='Output sizes of every layer')
    parser.add_argument('--regs', nargs='?', default='[0,0,0]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--train_loss', type=float, default=1,
                        help='Caculate training loss or not')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    return parser.parse_args()

def training(model, dataset, args):

    with tf.Session() as sess:
        # pretrain nor not
        sess.run(tf.global_variables_initializer())
        logging.info("initialized")
        print "initialized"
        writer = tf.summary.FileWriter('./graphs', sess.graph)

        # initialize for training batches
        if args.batch_gen == "fixed":
            if args.model == "FISM":
                samples = BatchItem.sampling(args, dataset, args.num_neg)
            else:
                samples = BatchUser.sampling(args, dataset, args.num_neg)

        # initialize for Evaluate
        if args.model == "FISM":
            EvalDict = EvalItem.init_evaluate_model(model, dataset)
        else:
            EvalDict = EvalUser.init_evaluate_model(model, dataset)

        # train by epoch
        for epoch_count in range(args.epochs):

            batch_begin = time()
            if args.model == "FISM":
                if args.batch_gen == "unfixed":
                    samples = BatchItem.sampling(args, dataset, args.num_neg)
                batches = BatchItem.shuffle(samples, args.batch_size, dataset)
            else :
                if args.batch_gen == "unfixed":
                    samples = BatchUser.sampling(args, dataset, args.num_neg)
                batches = BatchUser.shuffle(samples, args.batch_size)
            batch_time = time() - batch_begin
            train_begin = time()
            training_batch(model, sess, batches, args)
            train_time = time() - train_begin

            if epoch_count % args.verbose == 0:
                if args.train_loss:
                    loss_begin = time()
                    train_loss = training_loss(model, sess, batches, args)
                    loss_time = time() - loss_begin
                else:
                    loss_time, train_loss = 0, 0

                eval_begin = time()
                if args.model == "FISM":
                    hits, ndcgs, losses = EvalItem.eval(model, sess, dataset, EvalDict)
                else:
                    hits, ndcgs, losses = EvalUser.eval(model, sess, dataset, EvalDict)
                hr, ndcg, test_loss = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(losses).mean()
                eval_time = time() - eval_begin

                logging.info(
                    "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                        epoch_count, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time))
                print "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                    epoch_count, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time)

# input: batch_index (shuffled), model, sess, batches
# do: train the model optimizer
def training_batch(model, sess, batches, args):
    if args.model == 'FISM':
        user_input, num_idx, item_input, labels = batches
        for i in range(len(labels)):
            feed_dict = {model.user_input: user_input[i],
                         model.num_idx: num_idx[i][:, None],
                         model.item_input: item_input[i][:, None],
                         model.labels: labels[i][:, None]}
            sess.run(model.optimizer, feed_dict)
    else:
        user_input, item_input, labels = batches
        if args.loss_func == "logloss":
            for i in range(len(labels)):
                feed_dict = {model.user_input: user_input[i][:, None],
                             model.item_input: item_input[i][:, None],
                             model.labels: labels[i][:, None]}
        else:
            for i in range(len(labels)):
                feed_dict = {model.user_input: user_input[i][:, None],
                             model.item_input: item_input[i],
                             model.labels: labels[i][:, None]}
            sess.run(model.optimizer, feed_dict)

# input: model, sess, batches
# output: training_loss
def training_loss(model, sess, batches, args):
    train_loss = 0.0
    num_batch = len(batches[1])
    if args.model == 'FISM':
        user_input, num_idx, item_input, labels = batches
        for i in range(len(labels)):
            feed_dict = {model.user_input: user_input[i],
                         model.num_idx: num_idx[i][:, None],
                         model.item_input: item_input[i][:, None],
                         model.labels: labels[i][:, None]}
            train_loss += sess.run(model.loss, feed_dict)
    else:
        user_input, item_input, labels = batches
        if args.loss_func == "logloss":
            for i in range(len(labels)):
                feed_dict = {model.user_input: user_input[i][:, None],
                             model.item_input: item_input[i][:, None],
                             model.labels: labels[i][:, None]}
        else:
            for i in range(len(labels)):
                feed_dict = {model.user_input: user_input[i][:, None],
                             model.item_input: item_input[i],
                             model.labels: labels[i][:, None]}
        train_loss += sess.run(model.loss, feed_dict)

    return train_loss / num_batch

def init_logging(args):
    regs = eval(args.regs)
    path = "Log/%s/%s" % (args.dataset, args.model)
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(filename=path + "/log_pre%dembed_size%d_reg1%.7f_reg2%.7f_%s" % (
        args.pretrain, args.embed_size, regs[0], regs[1], strftime('%Y-%m-%d%H:%M:%S', localtime())),
                        level=logging.INFO)
    logging.info("begin training %s model ......" % args.model)
    logging.info("dataset:%s  pretrain:%d  embedding_size:%d"
                 % (args.dataset, args.pretrain, args.embed_size))
    logging.info("regs:%.8f, %.8f  learning_rate:%.4f  train_loss:%d"
                 % (regs[0], regs[1], args.lr, args.train_loss))

if __name__ == '__main__':

    # initialize logging
    args = parse_args()
    init_logging(args)

    #initialize dataset
    dataset = Dataset(args.path + args.dataset)

    #initialize models
    if args.model == "FISM" :
        model = FISM(dataset.num_items, args)
    elif args.model == "GMF" :
        model = GMF(dataset.num_users, dataset.num_items, args)
    elif args.model == "MLP" :
        model = MLP(dataset.num_users, dataset.num_items, args)
    model.build_graph()

    #start trainging
    training(model, dataset, args)
