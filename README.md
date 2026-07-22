# nanoMPG-v3-20225

### 20,225 parameter MLP
### 5 Hidden layers
### 14 input features
### Various activations (ReLU, Leaky ReLU, Sigmoid, GELU, SiLU)
### Choice of activation dependent on dropout metrics
### Learning rate scheduling by cosine annealing

#### To generate the dataset:

```bash
python3 data/dataworks.py
```

#### Dataset features: Actual physical interaction of features with realistic data generation through the use of probability distributions.

#### To train the model:

```bash
python3 train.py
```

#### In case training doesn't generate visuals:

```bash
python3 generate_visualisations.py
```

#### NOTE: mp4 generation requires ffmpeg to be on the system PATH.


### Inference is a todo for this project. The spec can be found in the documentation. 


### Authored by: Parv Sharma, FLAME University
