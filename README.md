# Detecting Injected Bias For Gender Prediction With Face Images

## Data
[UTKFace](https://susanqq.github.io/UTKFace/) under license for non-commercial research purposes only.

## Run one experiment with random seed = 0

### VGG16 model without pre-processing on fair training data (no bias injected)
```
src> python main.py nofair 0
```

### VGG16 model with FairBalanceClass pre-processing on fair training data (no bias injected)
```
src> python main.py fair 0
```

### VGG16 model with FairBalanceClass pre-processing on training data with injected bias favoring race = white
Degree of injected bias = 0.4
```
src> python main.py white 4 0
```

### VGG16 model with FairBalanceClass pre-processing on training data with injected bias favoring race = black
Degree of injected bias = 0.4
```
src> python main.py black 4 0
```

### Summarize results in the _result_ folder to generate a csv file under the _csv_ folder
```
src> python main.py summarize_result
```

## Submit 30 jobs at once using slurm
sbatch --array=1-30 fair.sh


