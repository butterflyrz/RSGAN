import math
import numpy as np
from Models.FISM import FISM
from Models.GMF import GMF
from Models.MLP import MLP

from time import time

class EvalFactory(object):
    @staticmethod
    def getEval(model, dataset):

        if isinstance(model, FISM):
            return ItemBasedEvaluation(model, dataset)

        elif isinstance(model, MLP) or isinstance(model, GMF) or isinstance(model, MF):
            return UserBasedEvaluation(model, dataset)

        else:
            return Evaluation(model, dataset)

class Evaluation(object):
    def __init__(self, model, dataset):
        self,DicList = None
        raise NotImplementedError

    def eval(self, model, sess, dataset):
        # calculate hr, ndcgs, losses
        hits, ndcgs, losses = [], [], []
        for idx in xrange(len(self.DictList)):
            (hr, ndcg, loss) = self._eval_one_rating(idx, dataset, model, sess)
            hits.append(hr)
            ndcgs.append(ndcg)
            losses.append(loss)
        return (hits, ndcgs, losses)

    def _eval_one_rating(self, idx, dataset, model, sess):
        self.predictions,self.loss = sess.run([model.output, model.loss], feed_dict = self.DictList[idx])

        self.neg_predict, self.pos_predict = self.predictions[:-1], self.predictions[-1] # the last is gtItem
        position = (self.neg_predict >= self.pos_predict).sum()

        hr = self.get_hr(position)
        ndcg = self.get_ndcg(position)
        loss = self.loss

        return (hr, ndcg, loss)

    def get_hr(self, position):
        return position < self.K

    def get_ndcg(self, position):
        return math.log(2) / math.log(position + 2) if position < self.K else 0


class ItemBasedEvaluation(Evaluation):
    def __init__(self, model, dataset):
        DictList = []
        for idx in xrange(len(dataset.testRatings)):
            user, gtItem = dataset.testRatings[idx]

            # if the negative sampling is fixed
            if model.eval == 'local':
                self.K = 10  # HR@10
                items = dataset.testNegatives[user]

            # else: all ranking mode
            else:
                self.K = 100  # HR@10
                items = set(range(dataset.num_items)) - set(dataset.trainList[user])
                if gtItem in items: items.remove(gtItem)
                items = list(items)

            # add the leave out one positive sample
            items.append(gtItem)
            item_input = np.array(items)[:, None]

            # initialize user_input
            users = dataset.trainList[idx]
            user_input = np.tile(np.array(users), (1, len(items)))

            # initialize num_idx
            num_idx_ = len(users)
            num_idx = np.full(len(items),num_idx_, dtype=np.int32 )[:,None]

            # initialize labels
            labels = np.zeros(len(items))[:, None]
            labels[-1] = 1

            # return the feed dict
            feed_dict = {model.user_input: user_input, model.num_idx: num_idx, model.item_input: item_input, model.labels: labels}
            DictList.append(feed_dict)
        print("already load the evaluate model...")
        self.DictList = DictList


class UserBasedEvaluation(Evaluation):
    def __init__(self, model, dataset):
        DictList = []
        for idx in xrange(len(dataset.testRatings)):
            user, gtItem = dataset.testRatings[idx]

            if model.eval == 'local':
                self.K = 10  # HR@10
                items = dataset.testNegatives[idx]
            else: # all ranking evaluation
                self.K = 100  # HR@10
                items = set(range(dataset.num_items)) - set(dataset.trainList[user])
                if gtItem in items: items.remove(gtItem)
                items = list(items)

            # add the leave out one positive sample
            items.append(gtItem)
            item_input = np.array(items)[:, None]

            # initialize user_input
            user_input = np.full(len(items), user, dtype=np.int32)[:, None]

            # initialize labels
            labels = np.zeros(len(items))[:, None]
            labels[-1] = 1

            feed_dict = {model.user_input: user_input, model.item_input: item_input, model.labels: labels}
            DictList.append(feed_dict)

        print("already load the evaluate model...")

        self.DictList = DictList
