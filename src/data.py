from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')
    
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
import torch

random_state=42
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '../input/AMZN_step2.csv')




def load_data(batch_size=64):
    df = pd.read_csv(data_path,  usecols=['close'], dtype={'close': 'float32'})
    close_price_reshaped = df.iloc[:, 0].to_numpy().reshape(-1, 1)
    minmax = MinMaxScaler().fit(close_price_reshaped) # Close index
    df_log = minmax.transform(close_price_reshaped) # Close index
    df_log = pd.DataFrame(df_log)
    # Generate the dataset
    data_transformed = create_training_data(df_log.loc[:, 0].to_list())
    # Define the number of test samples (last 60 days)
    num_test_samples = 60

    # Split the data based on this
    train_data = data_transformed[:-num_test_samples]  # All but the last 60 samples
    val_data = data_transformed[-num_test_samples:]   # The last 60 samples
    
    # Prepare the dataloaders
    val_dataset = StockDataset(train_data)
    test_dataset = StockDataset(val_data)
    
    train_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader



# Transform to list of tuples
def create_training_data(prices, window_size=30, target_size=1):
    data = []
    for i in range(len(prices) - window_size - target_size):
        x = prices[i:i + window_size]  # 30 days window
        y = [([1] if prices[i + window_size + j] > prices[i + window_size + j - 1] else [0]) for j in range(target_size)]
        data.append((np.array(x, dtype=np.float32), np.array(y, dtype=np.int64)))  # Convert booleans to float for compatibility with PyTorch
    return data

class StockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx][0]
        # because currently only for each day close price is used -> (30, 1)
        X = X.reshape(30, 1)
        y = self.data[idx][1]
        return X, y
    
    
if __name__ == '__main__':
    load_data()