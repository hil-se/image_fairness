from experiment import Experiment
from demos import cmd
import numpy as np
try:
  import cPickle as pickle
except:
  import pickle
import os
import pandas as pd

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

def white(ratio=4, seed=0):
    np.random.seed(int(seed))
    data_path = "../data/UTKFace"
    result_path = "../result/white_"+str(ratio)+"/"
    inject_ratio = {"race": [float(ratio)/10, -float(ratio)/10]}
    result = exp(data_path, fair=True, inject=inject_ratio)
    saveto = result_path + str(seed) + ".pickle"
    with open(saveto, "wb") as f:
        pickle.dump(result, f)

def nonwhite(ratio=4, seed=0):
    np.random.seed(int(seed))
    data_path = "../data/UTKFace"
    result_path = "../result/nonwhite_"+str(ratio)+"/"
    inject_ratio = {"race": [-float(ratio)/10, float(ratio)/10]}
    result = exp(data_path, fair=True, inject=inject_ratio)
    saveto = result_path + str(seed) + ".pickle"
    with open(saveto, "wb") as f:
        pickle.dump(result, f)

def white_no(ratio=4, seed=0):
    np.random.seed(int(seed))
    data_path = "../data/UTKFace"
    result_path = "../result/white_no_"+str(ratio)+"/"
    inject_ratio = {"race": [float(ratio)/10, -float(ratio)/10]}
    result = exp(data_path, fair=False, inject=inject_ratio)
    saveto = result_path + str(seed) + ".pickle"
    with open(saveto, "wb") as f:
        pickle.dump(result, f)

def nonwhite_no(ratio=4, seed=0):
    np.random.seed(int(seed))
    data_path = "../data/UTKFace"
    result_path = "../result/nonwhite_no_"+str(ratio)+"/"
    inject_ratio = {"race": [-float(ratio)/10, float(ratio)/10]}
    result = exp(data_path, fair=False, inject=inject_ratio)
    saveto = result_path + str(seed) + ".pickle"
    with open(saveto, "wb") as f:
        pickle.dump(result, f)

def summarize_result(path = "../result/", output = "../csv/result.csv"):
    treatments = os.listdir(path)
    compare = {}
    for treatment in treatments:
        results = None
        files = os.listdir(path+treatment)
        for file in files:
            with open(path+treatment+"/"+file, "rb") as f:
                result = pickle.load(f)
            if result['eod_pair'] == ((1,),(0,)):
                result['eod'] = - result['eod']
            if result['aod_pair'] == ((1,),(0,)):
                result['aod'] = - result['aod']
            result.pop('eod_pair', None)
            result.pop('aod_pair', None)
            if results == None:
                results = {key:[] for key in result}
            results = {key:results[key]+[result[key]] for key in result}
        compare[treatment] = results

    display = {"Treatment":[]}
    for treatment in treatments:
        display["Treatment"].append(treatment)
        medians = median_dict(compare[treatment])
        for key in medians:
            if key not in display:
                display[key] = []
            display[key].append(medians[key])
    df = pd.DataFrame(display)
    df.to_csv(output, index=False)

def median_dict(results, use_iqr = True):
    # Compute median value of lists in the dictionary
    for key in results:
        if type(results[key]) == dict:
            results[key] = median_dict(results[key], use_iqr = use_iqr)
        else:
            med = np.median(results[key])
            if use_iqr:
                iqr = np.percentile(results[key],75)-np.percentile(results[key],25)
                results[key] = "%d (%d)" % (med*100, iqr*100)
            else:
                results[key] = "%d" % (med*100)
    return results

if __name__ == "__main__":
    eval(cmd())
