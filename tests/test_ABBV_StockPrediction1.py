import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from omegaconf import DictConfig

# Mock configuration for testing
mock_cfg = DictConfig({
    "model": {
        "parameters": {
            "learning_rate": 0.001,
            "epochs": 10,
            "batch_size": 32
        },
        "hidden_dim": 50,
        "dropout": 0.2
    },
    "dataset": {
        "path": "src/data/stock_data/ABBV.csv"
    },
    "training": {
        "look_back": 60,
        "early_stopping_patience": 3,
        "validation_split": 0.2
    }
})

# Sample data for testing
sample_data = {
    'date': pd.date_range(start='1/1/2020', periods=100, freq='D'),
    'open': np.random.rand(100),
    'high': np.random.rand(100),
    'low': np.random.rand(100),
    'close': np.random.rand(100)
}
sample_df = pd.DataFrame(sample_data)
sample_csv_path = 'tests/sample_data.csv'
sample_df.to_csv(sample_csv_path, index=False)

@patch('src.models.logging.config.logger')
@patch('wandb.init')
@patch('wandb.config')
@patch('wandb.log')
@patch('pandas.read_csv')
def test_main(mock_read_csv, mock_wandb_log, mock_wandb_config, mock_wandb_init, mock_logger):
    # Mock read_csv to return the sample dataframe
    mock_read_csv.return_value = sample_df

    from src.models.ABBV_StockPrediction1 import main  # Import the main function from your script

    # Run the main function with the mock configuration
    with patch('hydra.utils.get_original_cwd', return_value=os.getcwd()):
        main(mock_cfg)

    # Add assertions here to validate the behavior of your main function
    assert mock_logger.info.called
    assert mock_wandb_init.called
    assert mock_read_csv.called
    assert mock_wandb_log.called
    assert mock_wandb_config.learning_rate == mock_cfg.model.parameters.learning_rate
    assert os.path.exists('reports/stock_price_prediction.png')

    # Clean up the created sample data file
    if os.path.exists(sample_csv_path):
        os.remove(sample_csv_path)
