# Neural networks

### Quick start

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/seg/tutorials-2018/master?filepath=1808_Neural_networks%2FManuscript.ipynb)


### Build your own environment

This tutorial serves as a deep look at the simplest of neural networks, the multilayer perceptron. The [manuscript notebook](https://github.com/seg/tutorials-2018/blob/master/1808_Neural_Network/manuscript.ipynb) uses no dependencies other than [Numpy](http://www.numpy.org/) to build up a neural network, train it to approximate angle-dependent seismic reflectivity, and then predict values inside (and outside) the domain of interest.

There's a code block which uses [tqdm](https://pypi.org/project/tqdm/) to render a progress bar to screen during training. To install tqdm:

    pip install tqdm

And to plot the results you'll need [matplotlib](https://matplotlib.org/2.0.0/users/installing.html):

    pip install matplotlib

To run the [load_and_process_data.ipynb](load_and_process_data.ipynb) you will need [welly](https://github.com/agile-geoscience/welly) and [bruges](https://github.com/agile-geoscience/bruges)...

    pip install welly bruges
    
...and [scikit-learn](http://scikit-learn.org/stable/index.html):

    pip install scikit-learn
    
It's a good idea to make a virtial environment for your projects. You can easily do this with `conda`:

    conda env create -f environment.yml


### So Long and Thanks for All the Fish

* Thanks to [Expero](https://experoinc.com) for encouraging me to continue to publish.
* Thanks to [Matt Hall](https://agilescientific.com/who/) for continuously pushing our professional community to modernize with efforts like the one you're reading now.
* Thanks to Lukas Mosser and Jesper Dramsch for useful comments on an earlier version of this manuscript.
