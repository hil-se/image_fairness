from matplotlib import image
from os import listdir
from os.path import isfile, join
from collections import Counter
from cnn import CNN, VGG16, VGG
import numpy as np
from pdb import set_trace

class Experiment:
    def __init__(self, data_path="../data/UTKFace", label = "sex", protected_attrs = ["race"]):
        self.data = self.load_data(data_path)
        self.X = self.data["image"]
        self.y = self.data[label]
        self.protected = protected_attrs
        self.model = VGG()
        # self.target = the minority class
        stat = Counter(self.y)
        self.target = list(stat.keys())[np.argmin(list(stat.values()))]
        self.inject_ratio = None

    def load_data(self, data_path="../data/UTKFace"):
        figures = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f.split('.')[-1] == 'jpg']
        data = {'image': [], 'age': [], 'sex': [], 'race': []}
        for f in figures:
            info = f.split('.')[0].split('_')
            race = int(info[2])
            if race > 1:
                continue
            data['race'].append(race)
            data['age'].append(int(info[0]))
            data['sex'].append(int(info[1]))
            data['image'].append(image.imread(join(data_path, f)))
        data = {key: np.array(data[key]) for key in data}
        print(Counter(data['race']))
        print(Counter(data['sex']))
        return data

    def FairBalance(self, train, y, class_balance = True):
        sample_weight = [1.0]*len(train)
        group = {}
        for i, id in enumerate(train):
            g = tuple([self.data[p][id] for p in self.protected]+[y[i]])
            if g not in group:
                group[g] = []
            group[g].append(i)
        class_weight = Counter(y)
        if class_balance:
            class_weight = {key: 1.0 for key in class_weight}
        for g in group:
            weight = class_weight[g[-1]] / len(group[g])
            for i in group[g]:
                sample_weight[i] = weight
        sample_weight = np.array(sample_weight) * len(train) / sum(sample_weight)
        return sample_weight


    def split(self, N, ratio=0.7):
        size = int(N*ratio)
        train = np.random.choice(N, size, replace=False)
        test = np.array(list(set(range(N))-set(train)))
        return train, test

    def evaluate(self, test):
        pred = self.model.predict(self.X[test])
        group = {}
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i, id in enumerate(test):
            g = tuple([self.data[p][id] for p in self.protected])
            if g not in group:
                group[g] = {"tp":0, "fp":0, "tn":0, "fn":0}
            if self.y[id] == self.target:
                if pred[i] == self.target:
                    tp += 1
                    group[g]["tp"] += 1
                else:
                    fn += 1
                    group[g]["fn"] += 1
            else:
                if pred[i] == self.target:
                    fp += 1
                    group[g]["fp"] += 1
                else:
                    tn += 1
                    group[g]["tn"] += 1
        result = {}
        result["acc"] = float(tp+tn) / (tp+fp+tn+fn)
        prec = float(tp) / (tp+fp)
        tpr = float(tp) / (tp+fn)
        result["f1"] = 2 * float(prec * tpr) / (prec + tpr)
        for g in group:
            group[g]["tpr"] = float(group[g]["tp"]) / (group[g]["tp"]+group[g]["fn"])
            group[g]["tfpr"] = 0.5 * (group[g]["tpr"] + float(group[g]["fp"]) / (group[g]["fp"] + group[g]["tn"]))
        max_eod = 0
        max_aod = 0
        eod_pair = None
        aod_pair = None
        for g1 in group:
            for g2 in group:
                eod = group[g1]["tpr"] - group[g2]["tpr"]
                aod = group[g1]["tfpr"] - group[g2]["tfpr"]
                if eod > max_eod:
                    max_eod = eod
                    eod_pair = (g1, g2)
                if aod > max_aod:
                    max_aod = aod
                    aod_pair = (g1, g2)
        result["eod"] = max_eod
        result["aod"] = max_aod
        result["eod_pair"] = eod_pair
        result["aod_pair"] = aod_pair
        return result

    def inject_bias(self, inject_ratio):
        # inject_ratio={attribute1: [ratio11, ratio12], attribute2: [ratio21, ratio22], ...}
        # ratio11 = 0.2 , ratio12 = -0.1 then
        #   1. 20% of (attribute1 =0 AND label = nontarget) will be changed to label = target,
        #   2. 10% of (attribute1 =1 AND label = target) will be changed to label = nontarget.
        # ratio21 = -0.1 , ratio22 = 0.2 then
        #   1. 10% of (attribute2 =0 AND label = target) will be changed to label = nontarget,
        #   2. 20% of (attribute1 =1 AND label = nontarget) will be changed to label = target.
        self.inject_ratio = inject_ratio

    def inject(self, sample):
        self.injected = []
        non_target = list(set(Counter(self.y).keys())-{self.target})[0]
        biased_y = self.y[sample]
        # perform bias injection on the input data.
        for attribute in self.inject_ratio:
            # try:
            #     ind = self.protected.index(attribute)
            # except:
            #     print("Error: Attribute %s does not exist in the protected attributes." %attribute)
            #     sys.exit(1)
            if attribute not in self.protected:
                print("Error: Attribute %s does not exist in the protected attributes." % attribute)
                break
            else:
                for group, ratio in enumerate(self.inject_ratio[attribute]):
                    to_change = non_target if ratio>0 else self.target
                    change_to = self.target if ratio>0 else non_target
                    change = np.where((self.data[attribute][sample] == group) & (self.y[sample]==to_change))[0]
                    size = int(np.abs(ratio)*len(change))
                    selected = np.random.choice(change, size, replace=False)
                    self.injected.extend(list(selected))
                    for i in selected:
                        biased_y[i] = change_to
        return biased_y

    def exp(self, fairbalance=True):
        sample_weight = None
        train, test = self.split(len(self.y))
        y = self.inject(train) if self.inject_ratio != None else self.y[train]
        if fairbalance:
            sample_weight = self.FairBalance(train, y, class_balance=True)
        self.model.fit(self.X[train], y, sample_weight=sample_weight)
        result = self.evaluate(test)
        print(result)
        return result
