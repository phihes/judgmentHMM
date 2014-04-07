Modeling user judgments with HMM
==============================

Includes four different scripts, examples (in /src/) and data.
___


## Scripts (/src/)


**judgmentHMM.py** - A small framework to create and validate different models of judgments. Provides easy setup of parameter sweeps, model performance metrics and model comparison. Data are passed as CSV. The included class *Experiment* allows to set up and run one or a series of experiments (creating a model with certain parameters, validating the model, calculating the performance metrics). Results are saved as csv, where each line corresponds to one experiment / parameters setting. See */docs/index.html* and the source for more information.

**gmmhmm.py** - Train a hidden Markov model with Gaussian mixture models created from turn features' principal components. Returns cross-validation scores / performance metrics, see judgmentHMM.TestResults.

**LR.py** - Linear regression using the data's principal components.

**PCA.py** - Script used to generate the principal components in the data.

**hmmParams.py** - Calculation of HMM parameters, used in gmmhmm.py


## Documentation (/docs/index.html)

Auto-generated documentation. See source for more comments.

## Data (/data/)

Pre-processed data from experiment, in CSV format.
