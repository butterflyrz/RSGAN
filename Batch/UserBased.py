import numpy as np
from multiprocessing import Pool

from Models import FISM, GMF, MLP

_user_input = None
_item_input = None
_labels = None
_batch_size = None
_index = None
_dataset = None

def _get_train_batch(i):
    user_batch, item_batch, labels_batch = [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input[_index[idx]])
        labels_batch.append(_labels[_index[idx]])
    return np.array(user_batch), np.array(item_batch), np.array(labels_batch)


class UserBasedBatchGen(object):
    def __init__(self, args, dataset):

        self.batch_size = args.batch_size
        self.dataset = dataset

        _user_input, _item_input, _labels = [], [], []
        num_negatives = args.num_neg
        num_users, num_items =  dataset.trainMatrix.shape
        if args.loss_func == "BPR":
            for (u, i) in dataset.trainMatrix.keys():
                # positive instance
                item_pair = []
                _user_input.append(u)
                item_pair.append(i)
                _labels.append(1)
                # negative instances
                j = np.random.randint(num_items)
                while dataset.trainMatrix.has_key((u, j)):
                    j = np.random.randint(num_items)
                item_pair.append(j)
                _item_input.append(item_pair)
        else:
            for (u, i) in dataset.trainMatrix.keys():
                # positive instance
                _user_input.append(u)
                _item_input.append(i)
                _labels.append(1)
                # negative instances
                for t in xrange(num_negatives):
                    j = np.random.randint(num_items)
                    while dataset.trainMatrix.has_key((u, j)):
                        j = np.random.randint(num_items)
                    _user_input.append(u)
                    _item_input.append(j)
                    _labels.append(0)

        self.user_input, self.item_input, self.labels = \
                                _user_input, _item_input, _labels

    def shuffle(self):
        global _user_input
        global _item_input
        global _labels
        global _batch_size
        global _index
        _user_input, _item_input, _labels = \
            self.user_input, self.item_input, self.labels
        _batch_size = self.batch_size
        _index = range(len(_labels))
        np.random.shuffle(_index)
        self.num_batch = len(_labels) // _batch_size
        pool = Pool()
        res = pool.map(_get_train_batch, range(self.num_batch))
        pool.close()
        pool.join()
        user_list = [r[0] for r in res]
        item_list = [r[1] for r in res]
        labels_list = [r[2] for r in res]
        self.batches = (user_list, item_list, labels_list)

    def feed(self, model, i):
        user_input, item_input, labels = self.batches
        feed_dict = {model.user_input: user_input[i][:, None],
                     model.item_input: item_input[i][:, None],
                     model.labels: labels[i][:, None]}
        return feed_dict