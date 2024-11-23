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
data_path = os.path.join(dir_path, '../input/data_sampled.csv')




def load_data(batch_size=64):
    df = pd.read_csv('input/AMZN_step2.csv',  usecols=['close'], dtype={'close': 'float32'})
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

def convert_array_string_numpy(array_str):
    return np.fromstring(array_str.strip('[]'), sep=' ', dtype=np.float32).tolist()


class StockDataset(Dataset):
    def __init__(self, X_df, Y):
        self.X_etc = X_df.drop(columns="combined_text").to_numpy(dtype=np.float32)
        self.X_emb = np.array(X_df.combined_text.to_list(), dtype=np.float32)
        self.Y = np.argmax(Y, axis=1)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        X_etc = self.X_etc[idx]
        X_emb = self.X_emb[idx]
        # because currently only for each day close price is used -> (30, 1)
        y = self.Y[idx]
        return X_etc, X_emb, y
    
    # Or more automatically:
    def get_class_weights(self):
        counts = np.bincount(self.Y)
        n_samples = len(self.Y)
        n_classes = len(counts)
        weights = n_samples / (n_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)
    
    
if __name__ == '__main__':
    load_data()