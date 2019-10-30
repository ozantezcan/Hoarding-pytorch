# Clutter Image Rating - pytorch

Pytorch implementation of [Automatic Assessment of Hoarding Clutter from Images Using Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/8470375). 

### 1)Training

1. Required libraries are listed in ```requirements.txt```. Run ```pip install -r requirements.txt``` using ```python 3.6``` to install them. 
2. Follow the steps in ```train.ipynb```

### 2) Inference

1. Required libraries are listed in ```requirements_inf.txt```. Run ```pip install -r requirements_inf.txt``` using ```python 3.6``` to install them. 
2. Change ```model_path``` and ```image_path``` in ```inference.py``` with the desired paths.
3. Run ```python inference.py``` to see the estimated CIR score of the image



