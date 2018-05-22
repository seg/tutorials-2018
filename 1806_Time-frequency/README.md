# Time-frequency decomposition

This tutorial looks at the basics of time-frequency decomposition of time series such as geophysical signals. Related notebooks also cover the continuous wavelet transform (CWWT) and empirical mode decomposition (EMD). Finally, I include code for reproducing some 'benchmark' time series for exploring and testing decomposition algorithms.

These are the interesting files:

- [`Manuscript.ipynb`](Manuscript.ipynb) &mdash; the main notebook, containing the text and code from the manuscript. Start here.
- [`EMD.ipynb`](EMD.ipynb) &mdash; a quick look at EMD, a time series analysis method in the time domain.
- [`CWT.ipynb`](CWT.ipynb) &mdash; a quick look at the continuous wavelet transform, which decomposes the signal into scale vs time.
- [`Natural_signals.ipynb`](Natural_signals.ipynb) &mdash; this notebook extracts a series of natural, real-world digital signals from various sources, including the piano, a bat, a volcano, and an earthquake.
- [`Synthetic_signals.ipynb`](Synthetic_signals.ipynb) &mdash; a work in progress, this notebook generates a series of artificial digital signals.

### Environment

I recommend using conda environments for Python development.

The main notebook, `manuscript.ipynb`, should run without installing anything except `scipy`, `matplotlib`, and (optionally), `seaborn`.

You will need to install some software:

    pip install bruges

You also need to install PyEMD:

    git clone https://github.com/laszukdawid/PyEMD.git
    cd PyEMD
    python setup.py sdist
    pip install dist/EMD-signal-0.2.4.tar.gz

The notebooks `Natural_signals.ipynb` requires `obspy`:

    conda config --add channel conda-forge
    conda install obspy


  ### The manuscript

  The file `Manuscript.ipynb` contains the manuscript, and all of the code necessary to reproduce it.

  Some of the notebook cells contain _tags_ which control the behaviour of `nbconvert`, the Jupyter command which parses the notebook into other formats. The tags work like so:

  - `hide`: hide this cell completely, as well as any output it generates. (I generally don't include import statements in the manuscript, for example.)
  - `hidein`: hide the input, but keep the output. This includes the figures, but not all the code that's required to generate them. (`matplotlib` can be rather verbose.)
  - `hideout`: keep the input, but hide the output.

  The following steps &mdash; which require you to install `pandoc` and, if you want PDF output, `LaTeX` too &mdash; will first execute the notebook, which saves all of the figures as PNG files, then generate DOCX and PDF manuscript documents from the notebook:

      jupyter nbconvert --execute Manuscript.ipynb --to notebook --output Manuscript.ipynb

      jupyter nbconvert Manuscript.ipynb --to markdown --output Manuscript.md \
      --TagRemovePreprocessor.remove_input_tags={\"hidein\"} \
      --TagRemovePreprocessor.remove_all_outputs_tags={\"hideout\"} \
      --TagRemovePreprocessor.remove_cell_tags={\"hide\"}

      pandoc Manuscript.md -o Manuscript.pdf
      pandoc Manuscript.md -o Manuscript.docx

  Note, if Jupyter complains about `NoSuchKernel`, add your current environment to the kernelspec, e.g.:

      python -m ipykernel install --name myenv
