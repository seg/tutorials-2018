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
    
It's a good idea to make a virtual environment for your projects. You can easily do this with `conda`:

    conda env create -f environment.yml
    
### Update: derivative of the logistic function

In the published version of the article, the derivative of the activation function (i.e., the logistic function _sigma(z) = 1 / (1+exp(-z))_ ) was expressed as _sigma'(z) = z(1-z)_. Using a particular value makes it clear that this expression of the derivative is wrong (_z=0; z(1-z)[0]=0_ but the tangent of the sigmoid function is not horizontal on z=0).

One can show that _sigma'(z) = -exp(-x)/(1+exp(-x))^2 = sigma(z)*(1-sigma(z))_. A more detailed demonstration can be found there: (https://en.wikipedia.org/wiki/Logistic_function#Derivative). The expression of the derivative should be corrected from _z*(1-z)_ to _sigma(z)*(1-sigma(z)_ in the second equation of the paper. 

However, the python expression of the backward sigmoid function _(x*(1-x))_ makes it possible to compute the forward and the backward values while computing the exponential value only once. This uses the following composition: 

 * _a1 = sigma(z)_ (estimate exonential)
 * _derivative = sigma(a1,False) = sigma(a1)*(1-sigma(a1)) = sigma'(a1)_.

The code provided in the paper is thus correct. It is probably faster than an implementation that would compute independently the sigmoid function and it's derivative.

### So Long and Thanks for All the Fish

* Thanks to [Expero](https://experoinc.com) for encouraging me to continue to publish.
* Thanks to [Matt Hall](https://agilescientific.com/who/) for continuously pushing our professional community to modernize with efforts like the one you're reading now.
* Thanks to Lukas Mosser and Jesper Dramsch for useful comments on an earlier version of this manuscript.
