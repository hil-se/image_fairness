from matplotlib import image
from os import listdir
from os.path import isfile, join
from collections import Counter
from cnn import CNN
import numpy as np
from pdb import set_trace

class Experiment:
    def __init__(self, data_path="../data/UTKFace", label = "sex", protected_attrs = ["race"]):
        self.data = self.load_data(data_path)
        self.X = self.data["image"]
        self.y = self.data[label]
        self.protected = protected_attrs

    def load_data(self, data_path="../data/UTKFace"):
        figures = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f.split('.')[-1] == 'jpg']
        data = {'image': [], 'age': [], 'sex': [], 'race': []}
        for f in figures:
            info = f.split('.')[0].split('_')
            data['age'].append(int(info[0]))
            data['sex'].append(int(info[1]))
            data['race'].append(int(info[2]))
            data['image'].append(image.imread(join(data_path, f)))
        data = {key: np.array(data[key]) for key in data}
        print(Counter(data['race']))
        print(Counter(data['sex']))
        return data

    def split(self, N, ratio=0.7):
        size = int(N*ratio)
        train = np.random.choice(N, size, replace=False)
        test = np.array(list(set(range(N))-set(train)))
        return train, test

    def exp(self):
        train, test = self.split(len(self.y))
        model = CNN()
        model.fit(self.X[train], self.y[train], batch_size=128, epochs=10)
        pred = model.predict(self.X[test])

