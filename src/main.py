from experiment import Experiment
from demos import cmd


def exp(data_path = "../data/UTKFace"):
    experiment = Experiment(data_path)
    experiment.exp()



if __name__ == "__main__":
    eval(cmd())
