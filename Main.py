import os

import tensorflow as tf
import numpy as np

import logging

from time import time
from time import strftime
from time import localtime

from Models.ModelFactory import ModelFactory
from saver import SaverFactory
from Batch.BatchFactory import BatchFactory
from Evaluation import EvalFactory

from Dataset import Dataset

import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args():
    parser = argparse.ArgumentParser(description="Run RSGAN.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--model', nargs='?', default='GMF.py',
                        help='Choose model: GMF.py, MLP, FISM')
    parser.add_argument('--loss_func', nargs='?', default='logloss',
                        help='Choose loss: logloss, BPR')
    parser.add_argument('--eval', nargs='?', default='local',
                        help='Choose evaluation: local, global')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[32,16,8]',
                        help='Output sizes of every layer')
    parser.add_argument('--regs', nargs='?', default='[0,0,0]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--task', nargs='?', default='',
                        help='Add the task name for launching experiments')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--train_loss', type=float, default=1,
                        help='Caculate training loss or not')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    return parser.parse_args()

def training(model, dataset, args, saver = None): # saver is an object to save pq

    with tf.Session() as sess:
        # pretrain nor not
        sess.run(tf.global_variables_initializer())
        logging.info("initialized")
        print "initialized"

        # initialize for training batches
        batch_gen = BatchFactory.getBatchGen(model, args, dataset)

        # initialize for Evaluate
        evaluation = EvalFactory.getEval(model, dataset)

        # train by epoch
        for epoch_count in range(args.epochs):

            batch_begin = time()
            batch_gen.shuffle()
            batch_time = time() - batch_begin

            train_begin = time()
            training_batch(model, sess, batch_gen)
            train_time = time() - train_begin

            if epoch_count % args.verbose == 0:
                loss_begin = time()
                train_loss = training_loss(model, sess, batch_gen) if args.train_loss else 0
                loss_time = time() - loss_begin

                eval_begin = time()
                hits, ndcgs, losses = evaluation.eval(model, sess, dataset)
                hr, ndcg, test_loss = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(losses).mean()
                eval_time = time() - eval_begin

                logging.info(
                    "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                        epoch_count, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time))
                print "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                        epoch_count, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time)

        if saver != None:
            saver.save(model, sess)

# input: batch_index (shuffled), model, sess, batches
# do: train the model optimizer
def training_batch(model, sess, batch_gen):
    for i in range(batch_gen.num_batch):
        feed_dict = batch_gen.feed(model, i)
        sess.run(model.optimizer, feed_dict)

# input: model, sess, batches
# output: training_loss
def training_loss(model, sess, batch_gen):
    train_loss = 0.0
    for i in range(batch_gen.num_batch):
        feed_dict = batch_gen.feed(model, i)
        train_loss += sess.run(model.loss, feed_dict)
    return train_loss / batch_gen.num_batch

def init_logging(args):
    regs = eval(args.regs)
    path = "Log/%s_%s/" % (strftime('%Y-%m-%d_%H', localtime()), args.task)
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(filename=path + "%s_%s_log_pre%dembed_size%d_reg1%.7f_reg2%.7f%s" % (
        args.dataset, args.model, args.pretrain, args.embed_size, regs[0], regs[1],strftime('%Y_%m_%d_%H_%M_%S', localtime())),
                        level=logging.INFO)
    logging.info(args)
    print args

if __name__ == '__main__':

    # initialize logging
    args = parse_args()
    init_logging(args)

    #initialize dataset
    dataset = Dataset(args.path + args.dataset)

    #initialize models
    model = ModelFactory.getModel(args.model, dataset.num_users, dataset.num_items, args)
    model.build_graph()

    # start training
    training(model, dataset, args)

