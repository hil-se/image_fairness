from matplotlib import image
from os import listdir
from os.path import isfile, join

def load_data(data_path = "../data/UTKFace"):
    figures = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f.split('.')[-1]=='jpg']
    data = {'image': [], 'age': [], 'sex': [], 'race': []}
    for f in figures:
        info = f.split('.')[0].split('_')
        data['age'].append(info[0])
        data['sex'].append(info[1])
        data['race'].append(info[2])
        data['image'].append(image.imread(join(data_path, f)))
    return data