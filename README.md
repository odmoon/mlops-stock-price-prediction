MLOPS solution for stock price prediction model 
==============================
## Section 1 Project Proposal

## 1.1 Project scope and objectives

## Project overview
This project aims to develop a stock model prediction system that 
leverages Machine Learning (ML) to accurately predict stock prices. 
The core of this project is built on Python and utilizes ML model to 
analyze historical stock data and predict future trends. 
This project is enhanced with a complete MLOps solution for 
development, deployment, and monitoring of the ML models.

## Features 

* Data Processing: Automated scripts to fetch, clean, and prepare historical stock data for training.
* Model Training: Implementation of multiple ML models to evaluate their performance on stock price prediction.
* Model Evaluation: Rigorous testing and validation strategies to ensure the accuracy and reliability of the models.
* Model Deployment: Automated deployment of the best-performing model to a production environment.
* Continuous Integration and Continuous Deployment (CI/CD): CI/CD pipelines to automate the testing and deployment processes.
* Monitoring: Tools to monitor the model's performance in production and trigger retraining if necessary.

## Technology Stack

Python: Primary programming language for model development and scripting.
NumPy: For data manipulation and numerical processing.
and others 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## 1.2 Selection of Data and 1.4 Open-Source Tools

The dataset we chose is from Kaggle, here is the link: 
https://www.kaggle.com/datasets/svaningelgem/nyse-100-daily-stock-prices

The reason we chose this dataset to approach our objective of developing a model to predict stock prices accurately is it contains data from the Top 100 stocks in the market on the NYSE from January 1962-May 2024. With over 62 years of data, and the simple language involved in the dataset, we knew we could utilize this to develop our model. The dataset itself contains 100 csv files of different stocks, each file containing the date, and OHLC. OHLC refers to open, high, low and close which refers to the price at which transactions are completed, the highest and lowest transaction prices, and the final transaction price.

The preprocessing steps required to make it usable for our project are... (write more)


## 1.3 Model Considerations

The model architectures that are appropriate for our dataset that we are considering are LSTM (long short-term memory), ARIMA, and RNN (recurrent neural networks). This is because stock predictions require time-series prediction models... (Write more once decided which model)


## Members of Group Project and Roles
## Odonchimeg Bold
    - Setup repository on GitHub with cookiecutter template
    - wrote up the project scope and objective in README.md
    set up the environment as well with cookiecutter template to provide requirements.txt and respective python files
    - (Write more..)
## Maheen Khan
    - Added sections 1.2-1.4 to the README.md where the selection of dataset is justified and possible model considerations
    - Wrote up team members roles in README.md
    - (Write more..)
## Dylan Neal
    - Helped look for datasets and is working on the source code for algorithms for possible model architectures
    - (Write more..)
