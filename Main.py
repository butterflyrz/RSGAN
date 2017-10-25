import tensorflow as tf
import numpy as np

import logging

from time import time
from time import strftime
from time import localtime

from Models import FISM
from Models import GMF
from Models import MLP

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NAIS.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='pinterest-20',
                        help='Choose a dataset.')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with updating FISM variables, 2:Pretrain with fixed FISM variables.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_choice', nargs='?', default='user',
                        help='user: generate batches by user, fixed:batch_size: generate batches by batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--weight_size', type=int, default=8,
                        help='weight size.')
    parser.add_argument('--embed_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--data_alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--regs', nargs='?', default='[1e-7,1e-7,0]',
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Index of coefficient of embedding vector')
    parser.add_argument('--train_loss', type=float, default=1,
                        help='Caculate training loss or not')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Index of coefficient of sum of exp(A)')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--activation', type=int, default=0,
                        help='Activation for ReLU, sigmoid, tanh.')
    parser.add_argument('--algorithm', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')
    parser.add_argument('--logAttention', type=int, default=0,
                        help='log softmax attetnion or not')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='save the checkpoint or not')
    return parser.parse_args()

def training(model, dataset):

    with tf.Session() as sess:
        # pretrain nor not
        sess.run(tf.global_variables_initializer())
        logging.info("initialized")
        print "initialized"

        # initialize for training batches
        batch_begin = time()
        batches = data.shuffle(dataset, model.batch_choice, model.num_negatives)
        batch_time = time() - batch_begin

        num_batch = len(batches[1])
        batch_index = range(num_batch)

        # initialize the evaluation feed_dicts
        testDict = evaluate.init_evaluate_model(model, sess, dataset.testRatings, dataset.testNegatives,
                                                dataset.trainList)
        # train by epoch
        for epoch_count in range(model.epochs):

            train_begin = time()
            training_batch(batch_index, model, sess, batches)
            train_time = time() - train_begin

            if epoch_count % model.verbose == 0:

                if model.train_loss:
                    loss_begin = time()
                    train_loss = training_loss(model, sess, batches)
                    loss_time = time() - loss_begin
                else:
                    loss_time, train_loss = 0, 0

                eval_begin = time()
                (hits, ndcgs, losses, _) = evaluate.eval(model, sess, dataset.testRatings, dataset.testNegatives,
                                                         testDict)
                hr, ndcg, test_loss = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(losses).mean()
                eval_time = time() - eval_begin

                logging.info(
                    "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                        epoch_count, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time))
                print "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1fs] train_loss = %.4f [%.1fs]" % (
                    epoch_count, batch_time, train_time, hr, ndcg, test_loss, eval_time, train_loss, loss_time)

            batch_begin = time()
            batches = data.shuffle(dataset, model.batch_choice, num_negatives)
            np.random.shuffle(batch_index)
            batch_time = time() - batch_begin

# input: batch_index (shuffled), model, sess, batches
# do: train the model optimizer
def training_batch(batch_index, model, sess, batches):
    for index in batch_index:
        user_input, num_idx, item_input, labels = data.batch_gen(batches, index)
        feed_dict = {model.user_input: user_input, model.num_idx: num_idx[:, None],
                     model.item_input: item_input[:, None],
                     model.labels: labels[:, None]}
        sess.run(model.optimizer, feed_dict)

# input: model, sess, batches
# output: training_loss
def training_loss(model, sess, batches):
    train_loss = 0.0
    num_batch = len(batches[1])
    for index in range(num_batch):
        user_input, num_idx, item_input, labels = data.batch_gen(batches, index)
        feed_dict = {model.user_input: user_input, model.num_idx: num_idx[:, None],
                     model.item_input: item_input[:, None], model.labels: labels[:, None]}
        train_loss += sess.run(model.loss, feed_dict)
    return train_loss / num_batch


if __name__ == '__main__':
    args = parse_args()
    regs = eval(args.regs)

    logging.basicConfig(filename="Log/%s/log_pre%dembed_size%d_reg1%.7f_reg2%.7f_%s" % (
    args.dataset, args.pretrain, args.embed_size, regs[0], regs[1], strftime('%Y-%m-%d%H:%M:%S', localtime())),
                        level=logging.INFO)
    logging.info("begin training FISM model ......")
    logging.info("dataset:%s  pretrain:%d  embedding_size:%d"
                 % (args.dataset, args.pretrain, args.embed_size))
    logging.info("regs:%.8f, %.8f  learning_rate:%.4f  train_loss:%d"
                 % (regs[0], regs[1], args.lr, args.train_loss))

    dataset = Dataset(args.path + args.dataset)

    if args.model == "FISM" :
        model = FISM(dataset.num_items, args)
    elif args.model == "GMF" :
        model = GMF(dataset.num_items, args)
    elif args.model == "MLP" :
        model = FISM(dataset.num_items, args)

    model.build_graph()
    training(model, dataset)
