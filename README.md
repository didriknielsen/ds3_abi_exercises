# Exercises

Exercises for the Tutorial on Approximate Bayesian Inference at the Data Science Summer School 2018.

## Dependencies

We will use Python 3 for all exercises.
The following packages will be required:
* [`numpy`](https://www.scipy.org/install.html)
* `scipy` - see above.
* [`PyTorch`](https://pytorch.org/) - version 0.4.0 or greater.
* [`matplotlib`](https://matplotlib.org/users/installing.html)
* [`jupyter`](http://jupyter.org/install)
* [`ipywidgets`](https://ipywidgets.readthedocs.io/en/latest/user_install.html)

```
pip install -r requirements.txt
```

## Starting the Jupyter Notebook

From the repo's main directory, run:
```
jupyter notebook
```
This will automatically open a tab in your browser and display the exercise notebooks.

## Troubleshooting

If the interactive widgets do not work, try:

```
pip install ipywidgets --upgrade
jupyter nbextension enable --py widgetsnbextension
```
