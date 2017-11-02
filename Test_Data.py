import BatchGen.BatchGenUser as BatchUser
from Dataset import Dataset
import Evaluate.EvaluateUser as EvalUser

class TEST:
    def __init__(self):
        self.user_input = None
        self.item_input = None
        self.labels = None

dataset = Dataset('Data/ml-1m')
model = TEST
EvalDict = EvalUser.init_evaluate_model(model, dataset)
print EvalDict[0]['user_input'].shape