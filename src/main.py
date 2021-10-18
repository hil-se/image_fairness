from experiment import Experiment
from demos import cmd
import numpy as np
try:
  import cPickle as pickle
except:
  import pickle


def exp(data_path = "../data/UTKFace", fair = True, inject = None):
    experiment = Experiment(data_path)
    if inject != None:
        experiment.inject_bias(inject)
    result = experiment.exp(fairbalance = fair)
    return result

def nofair(seed = 0):
    np.random.seed(int(seed))
    data_path = "../data/UTKFace"
    result_path = "../result/nofair/"
    result = exp(data_path, fair=False)
    saveto = result_path + str(seed) + ".pickle"
    with open(saveto, "wb") as f:
        pickle.dump(result, f)

def fair(seed=0):
    np.random.seed(int(seed))
    data_path = "../data/UTKFace"
    result_path = "../result/fair/"
    result = exp(data_path, fair=True)
    saveto = result_path + str(seed) + ".pickle"
    with open(saveto, "wb") as f:
        pickle.dump(result, f)

def white(seed=0):
    np.random.seed(int(seed))
    data_path = "../data/UTKFace"
    result_path = "../result/white/"
    inject_ratio = {"race": [0.4, -0.4]}
    result = exp(data_path, fair=True, inject=inject_ratio)
    saveto = result_path + str(seed) + ".pickle"
    with open(saveto, "wb") as f:
        pickle.dump(result, f)

def black(seed=0):
    np.random.seed(int(seed))
    data_path = "../data/UTKFace"
    result_path = "../result/black/"
    inject_ratio = {"race": [-0.4, 0.4]}
    result = exp(data_path, fair=True, inject=inject_ratio)
    saveto = result_path + str(seed) + ".pickle"
    with open(saveto, "wb") as f:
        pickle.dump(result, f)

def exp_trained(data_path = "../data/UTKFace", checkpoint_filepath = './tmp/checkpoint'):
    experiment = Experiment(data_path)
    experiment.model.load_model(checkpoint_filepath)
    train, test = experiment.split(len(experiment.y))
    print(experiment.evaluate(test))
    return

if __name__ == "__main__":
    eval(cmd())
