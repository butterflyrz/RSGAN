from Models.FISM import FISM
from Models.MLP import MLP
from Models.GMF import GMF
from ItemBased import ItemBasedBatchGen
from UserBased import UserBasedBatchGen


def test(i):
    return i

# input: dataset(Mat, List, Rating, Negatives), batch_choice, num_negatives
# output: [_user_input_list, _item_input_list, _labels_list]

class BatchFactory(object):

    @staticmethod
    def getBatchGen(model, args, dataset):
        if isinstance(model, FISM):
            return ItemBasedBatchGen(args, dataset)
        elif isinstance(model, MLP) or isinstance(model, GMF):
            return UserBasedBatchGen(args, dataset)
        else:
            raise NotImplementedError