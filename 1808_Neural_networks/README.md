# Neural networks

This tutorial serves as a deep look at the simplest of neural networks, the multilayer perceptron. The [manuscript notebook](https://github.com/seg/tutorials-2018/blob/master/1808_Neural_Network/manuscript.ipynb) uses no dependencies other than [Numpy](http://www.numpy.org/) to build up a neural network, train it to approximate angle-dependent seismic reflectivity, and then predict values inside (and outside) the domain of interest.

There's a code block which uses [tqdm](https://pypi.org/project/tqdm/) to render a progress bar to screen during training. To install tqdm:

`pip install tqdm`

And to plot the results you'll need [matplotlib](https://matplotlib.org/2.0.0/users/installing.html):

`pip install matplotlib`

### So Long and Thanks for All the Fish

* Thanks to [Expero](https://experoinc.com) for encouraging me to continue to publish.
* Thanks to [Matt Hall](https://agilescientific.com/who/) for continuously pushing our professional community to modernize with efforts like the one you're reading now.
* Thanks to Lukas Mosser and Jesper Dramsch for useful comments on an earlier version of this manuscript.
