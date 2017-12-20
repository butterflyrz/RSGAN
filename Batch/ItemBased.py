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
    user_batch, num_batch, item_batch, labels_batch = [], [], [], []
    begin = i * _batch_size
    trainList = _dataset.trainList
    num_items = _dataset.num_items
    for idx in range(begin, begin + _batch_size):
        user_idx = _user_input[_index[idx]]
        item_idx = _item_input[_index[idx]]
        nonzero_row = []
        nonzero_row += trainList[user_idx]
        num_batch.append(_remove_item(num_items, nonzero_row, item_idx))
        user_batch.append(nonzero_row)
        item_batch.append(item_idx)
        labels_batch.append(_labels[_index[idx]])
    return np.array(_add_mask(num_items, user_batch, max(num_batch))), \
           np.array(num_batch), np.array(item_batch), np.array(labels_batch)

def _remove_item(feature_mask, users, item):
    flag = 0
    # delete item in item_input
    for i in range(len(users)):
        if users[i] == item:
            users[i] = users[-1]
            users.pop()
            flag = 1
            break
    return len(users) - flag

def _add_mask(feature_mask, features, num_max):
    # uniformalize the length of each batch
    for i in xrange(len(features)):
        features[i] = features[i] + [feature_mask] * (num_max + 1 - len(features[i]))
    return features


class ItemBasedBatchGen(object):
    def __init__(self, args, dataset):

        self.batch_size = args.batch_size
        self.dataset = dataset
        num_negatives = args.num_neg

        _user_input, _item_input, _labels = [], [], []
        num_users, num_items = dataset.trainMatrix.shape
        for (u, i) in dataset.trainMatrix.keys():
            # positive instance
            _user_input.append(u)
            _item_input.append(i)
            _labels.append(1)
            # negative instances
            for _ in xrange(num_negatives):
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
        global _dataset

        _user_input, _item_input, _labels = \
            self.user_input, self.item_input, self.labels
        _batch_size = self.batch_size
        _dataset = self.dataset
        _index = range(len(_labels))
        np.random.shuffle(_index)
        self.num_batch = len(_labels) // _batch_size
        pool = Pool()
        res = pool.map(_get_train_batch, range(self.num_batch))
        pool.close()
        pool.join()
        user_list = [r[0] for r in res]
        num_idx = [r[1] for r in res]
        item_list = [r[2] for r in res]
        labels_list = [r[3] for r in res]
        self.batches = (user_list, num_idx, item_list, labels_list)

    def feed(self, model, i):
        user_input, num_idx, item_input, labels = self.batches
        feed_dict = {model.user_input: user_input[i],
                     model.num_idx: num_idx[i][:, None],
                     model.item_input: item_input[i][:, None],
                     model.labels: labels[i][:, None]}
        return feed_dict


