from FISM import FISM
from GMF import GMF
from MLP import MLP

class ModelFactory(object):
    @staticmethod
    def getModel(model, num_users, num_items, args):
        if model == "GMF":
            return GMF(num_users, num_items, args)
        elif model == "MLP":
            return MLP(num_users, num_items, args)
        elif model == "FISM":
            return FISM(num_users, num_items, args)
        else:
            return Model(num_users, num_items, args)

