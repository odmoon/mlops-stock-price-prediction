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

## Scope

* Data Processing: Automated scripts to fetch, clean, and prepare historical stock data for training.
* Model Training: Implementation of multiple ML models to evaluate their performance on stock price prediction.
* Model Evaluation: Rigorous testing and validation strategies to ensure the accuracy and reliability of the models.
* Model Deployment: Automated deployment of the best-performing model to a production environment.
* Continuous Integration and Continuous Deployment (CI/CD): CI/CD pipelines to automate the testing and deployment processes.
* Monitoring: Tools to monitor the model's performance in production and trigger retraining if necessary.

## Prerequisites

To run this project, you'll need:
- Python 3.8+
- pip (Python package manager)

## Environment Setup

Follow these steps to set up your environment and run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/odmoon/mlops-stock-price-prediction.git
cd mlops-stock-price-prediction
```
### 2. Create and activate a Virtual Environment 
* Unix or MacOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
* Windows: 
```bash
python3 -m venv venv
.\venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt 
``` 
or manually install following: 
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow torch hydra-core wandb rich
```
### 4. Running the code 
```bash
python src/models/ABBV_StockPrediction1.py
```
### 5. Docker Instructions
- Download and Install docker from (https://docs.docker.com/get-docker/)

```bash
### Building the Docker Image  
docker build -t mlops-stock-price-prediction .
```
```bash
### Running the Docker Container
docker run -it --rm mlops-stock-price-prediction
```
### 6. Weights and Biases (wandb) Integration
This project uses Weights and Biases for experiment tracking and model evaluation. You can view the experiment results and logs on the wandb project page.

1. Ensure you have a wandb account. You can sign up on the wandb.ai webpage. 
2. Log in to wandb from your terminal:

```bash
### Copy your wandb API token and paste after the command below
wandb login
```
3. After running the model training script, the metrics, including loss and RMSE scores, are logged to wandb.
Note: Pass your WandB API token as environment variable in DockerFile if the model runs on container

## Shared wandb Report:
* https://api.wandb.ai/links/odmoon/bk64h1b3 
In case WandB trial ends. This is the sample output of the project experiment report. 


### 7. Logging 
The project uses Python's built-in logging module configured with rich for enhanced log formatting. The logging configuration is defined in src/models/logging/config.py. The logging configuration includes handlers for console output, info logs, and error logs. 
Logs are stored in src/logs directory. 

### 8. Hydra configuration 

Configuration files for Hydra stored in the conf directory at the root of the project. Config YAML file defines parameters for different aspects of the model, such as dataset paths, model architecture, training parameters. The output log is stored in outputs folder in root directory. 
Hydra allows you to override any configuration parameter from the command line. For example, to change the learning rate and the number of epochs, you can run:

```bash
python3 src/models/ABBV_StockPrediction1.py model.parameters.learning_rate=0.001 model.parameters.epochs=50
```

Project Organization
------------
    ├── conf
    │   └── config.yaml    <- Hydra config file for parameter configuration management
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train` /future task/
    ├── README.md          <- The top-level README for developers using this project.
    ├── data /will be used in future, for now static csv file is loaded from src/data/stock_data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
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
    │   │   └── stock_data
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── ABBV_StockPrediction1.py
    │   └── 
    └── 


--------

## 1.2 Selection of Data and 1.4 Open-Source Tools

The dataset we chose is from Kaggle, here is the link: 
https://www.kaggle.com/datasets/svaningelgem/nyse-100-daily-stock-prices

The reason we chose this dataset to approach our objective of developing a model to predict stock prices accurately is it contains data from the Top 100 stocks in the market on the NYSE from January 1962-May 2024. With over 62 years of data, and the simple language involved in the dataset, we knew we could utilize this to develop our model. The dataset itself contains 100 csv files of different stocks, each file containing the date, and OHLC. OHLC refers to open, high, low and close which refers to the price at which transactions are completed, the highest and lowest transaction prices, and the final transaction price.

The preprocessing steps required to make it usable for our project are... (write more)


## 1.3 Model Considerations

The model architectures that are appropriate for our dataset that we are considering are LSTM (long short-term memory), ARIMA, and RNN (recurrent neural networks). This is because stock predictions require time-series prediction models... (Write more once decided which model)

Here are the steps that we followed in relation to our data set. 
    1. The data is loaded into the program
    2. We Convert the date column into date time and set it as a pandas DataFrame index 
    3. We Choose the features from the dataset for model input 
    4. We Scale the feature from 0 to 1 
    5. We Define 'look_back' which looks at past data to inform itself for future predictions
    6. The create_dataset builds I/O datasets for the NN. 
    7. The dataset is split into training and testing data at 67% for training and 33% for testing
    8. Two LTSM Layers are defined: Dropout and Dense; prevents overfitting
    9. Set up Early Stopping, which halts validation if model does not improve. 
    10. The Model trains (fits) itself on the dataset
    11. Training and Testing both make predictions 
    12. The predictions revert to the original scale
    13. The RSME is calculated
    14. The date information is indexed
    15. The Model is plotted. 

## Profiling Documentation
# Profiling Documentation

## cProfile Profiling

### How to Run

1. Run the profiling script:
    ```bash
    python3 profile_cprofile.py
    ```

2. The results will be saved to `cprofile_results.txt`.

### Interpreting Results

- The results show function call counts and the cumulative time spent in each function.
- Focus on functions with high cumulative time for optimization.

### Optimizations

- **High time in data loading:** Consider using efficient data structures or libraries.
- **High time in training:** Optimize the model architecture or use more efficient training loops.

## Torch Profiler

### How to Run

1. Run the profiling script:
    ```bash
    python3 profile_torch_profiler.py
    ```

2. The results will be saved to `./log/torch_profiler`.

### Interpreting Results

- Use TensorBoard to visualize the profiling results:
    ```bash
    tensorboard --logdir=./log/torch_profiler
    ```

- Analyze the timeline, memory usage, and bottlenecks.

### Optimizations

- **High GPU usage:** Optimize batch sizes and model architecture.
- **Memory bottlenecks:** Use mixed precision training or gradient checkpointing.

## General Tips

- **Vectorization:** Ensure operations are vectorized and avoid Python loops in favor of NumPy operations.
- **Parallelism:** Use multi-threading or multi-processing for data loading and preprocessing.
- **Hardware Utilization:** Ensure the GPU is fully utilized during training.

## Model deployment and CI/CD 
# Github Action workflows
This project utilizes GitHub Actions to automate the CI/CD process. The workflows are defined in .github/workflows/ci.yml and include steps for testing, building Docker images, and running model training with Continuous Machine Learning (CML).

# CI Workflow
The CI workflow is triggered on every push and pull request to the main branch. 
It consists of the following jobs:
1. Test: Runs the unit tests (Pytest) and code quality checks with ruff.
2. Build Docker Image: Builds and pushes a Docker image to Docker Hub.
3. CML Run: Runs the model training and profiling, and publishes a report using CML.
Workflow File: .github/workflows/ci.yml

# Setup Instructions on Github action

1. Set up GitHub Secrets: Ensure the following secrets are set up in your GitHub repository:
DOCKER_PASSWORD: Docker Hub access token.
DOCKER_USERNAME: Docker Hub username.
WANDB_API_KEY: Weights and Biases API key passed along with the docker build and Dockerfile as env variable.
GITHUB_TOKEN: GitHub token for CML (automatically provided).
GOOGLE_APPLICATION_CREDENTIALS: Google Cloud service account JSON key /as the dvc is stored on remote GCP storage/ 

2. Every PR or merge on main action will trigger automatically. You can also manually trigger them from the Actions tab in your GitHub repository. Logs and actions for each workflow is stored in Actions tab on your github repo.  

## Report for findings, challenges, and areas for improvement

## Findings


We found that the model was able to predict very well in terms of data that it already knew, but was not confident with future, unknown, data. When attempting to predict the stock price one month past the dataset, it created a consistent downward vector that did not appear to be indicative of the stock's future value. We attribute this to the fact that the model may be overfitted and is not confident in predicting data that is foreign to what it already knows. 
As such, we are looking into other feature sets and possibilities to integrate into the model past pricing information like 'open' and 'high'. We believe that integrating sentiment analysis from news sources and social media, as well as larger economic indicators surrounding the stock, would be effective measures to train the model on as well in order to make accurate predictions. 
 

## Challenges
A challenge we had is finding a dataset we could understand in layman's terms that would be useful to predict. Even with finance in mind, datasets for stocks and cryptocurrency were either outdated or over complicated or too big or too small.

The main challenge we are encountering is prediction past the dataset. We can only seem to train model for the current dataset but its failing to do the prediction for the future stocks as it is showing a downward trend which is not realistic. Our model is over fitted for the dataset, so it is too reliant on the data and is not confident in predicting the data.

New Findings From Part 2:
I found profiling to be challenging but it turns out my main issues was not knowing knowledge like using Python3 instead of installing the long convulated way. I also could not get TensorBoard on local host to show my image, but I did get a log file generated for both cProfile and PyTorch. However, upon looking at the results, in Torch_Profiler, I could not make sense of what the results meant but I knew it was incorrect due to the localhost in TensorBoard not being able to show anything. For cProfile, it said it returned in 1 second with zero values, so I can tell there are still issues but I unfortunately ran out of time to resolve these issues. I wonder if it has to do with our stock prediction model not being accurate.

We also learned that a lot of the issues were not having a main function in the python file which we then added on and this resolved the profiling issues and the configuration management issue with Hydra. We were then able to generate a cprofile with actual results and time taken for the function, as well as the Hydra config.yaml file and showed the parameters needed. 
We also learned for Hydra we had to update the requirements.txt to show hydra-core==1.1.0 instead of just stating 'hydra'

Initally when we ran the Docker container, the graph was not saving automatically or showing, so to resolve this issue we improved the code to save the plot under the reports folder. We also do not want to expose API key in Docker, so instead of encrypting we set an environment variable so only the people who have access to the key can use it. 

## Areas for Improvement
Exploration for other predictive methods outside of our current dataset. Possibly integration of other datasets would show us a more clearer path to where the error lies. We want to focus on one part of the dataset to make sure the model is running correctly before implementing with over 100 files of data. 

A lot more practice with technologies such as cProfile, Torch, and Hydra would have been helpful. I think we had a huge dataset and complex model we were attempting and possibly working with a simpler model first in class would have been more helpful where we integrate all the technologies together instead of separately. Also I think a lot of issues we had was not knowing where certain files and folders go in the repo, whether its global or in src, so more practice with that would be helpful as well. We have a ton of logs that are produced and it slows down the model as well. 


## Members of Group Project and Roles
## Odonchimeg Bold
    - Setup repository on GitHub with cookiecutter template
    - wrote up the project scope, instructions of environment setup and dependencies in README.md
    set up the environment as well with cookiecutter template to provide requirements.txt and respective python files
    - Wrote up report summarizing findings, challenges encountered, and areas for improvement with team.
    - Containerized the model and tested locally using docker image build and container run. 
    - Applied Logging into main model along with rich handler and the configurations 
    - Integrated WandB experiment tracking and report dashboard with the metrics. 
    - Generated plot graphic into /reports directory

## Maheen Khan
    - Added sections 1.2-1.4 to the README.md where the selection of dataset is justified and possible model considerations
    - Wrote up team members roles in README.md
    - Updated README.md at end of part 2 with challenges and reports
    - Wrote up report summarizing findings, challenges encountered, and areas for improvement with team.
    - Wrote up profiling with cProfile and Torch.
## Dylan Neal
    - Proposed a preliminary long short-term memory model and wrote up the data documentation and training steps.
    - Converted jupyter notebook to source code for repo for submission
    - Wrote up report summarizing findings, challenges encountered, and areas for improvement with team.
    - Wrote up Hydra configuration in python file for stock prediction and config.yaml



## Sources Used 

https://www.datacamp.com/tutorial/lstm-python-stock-market

https://www.youtube.com/watch?si=4OR6BJm8pIQzUgHZ&v=YCzL96nL7j0&feature=youtu.be

https://docs.wandb.ai

ChatGPT 
GitHub Copilot
