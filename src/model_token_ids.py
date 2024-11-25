import torch.nn as nn
import numpy as np
import torch
from transformers import LlamaTokenizer, LlamaModel


class TweetsBlock(nn.Module):
    def __init__(self, 
                 dropout_rate=0.3):
        super(TweetsBlock, self).__init__()
        

        self.tweet_block = TweetBlock()
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming initialization often works better with ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, X_tweets_tkids):
        # Main path
        x = self.tweet_block(X_tweets_tkids)
        
        return x
    
    
class TweetBlock(nn.Module):
    def __init__(self, pretrained_model_name='ChanceFocus/finma-7b-nlp', 
                 context_len=40, 
                 output_dim=1024, 
                 dropout_rate=0.3):
        super(TweetBlock, self).__init__()
        
        # Load tokenizer and pre-trained model
        self.tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name)
        self.embedding_model = LlamaModel.from_pretrained(pretrained_model_name)
        
        # Freeze the embedding layer to use pre-trained embeddings
        for param in self.embedding_model.parameters():
            param.requires_grad = False
        
        # Extract embedding dimension from the model
        self.emb_dim = self.embedding_model.config.hidden_size
        
        # Define CNN layer
        self.cnn = nn.Conv1d(in_channels=self.emb_dim, 
                             out_channels=output_dim, 
                             kernel_size=3, 
                             stride=1, 
                             padding=1)  # Use padding to maintain sequence length
        
        # Define activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming initialization for CNN layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, X_tweet_tkids):
        """
        Args:
            X_tweet_tkids: Tensor of shape (batch_size, 30, 40), where:
                - batch_size: Number of days in the batch
                - 30: Number of time steps (days)
                - 40: Context length (tokens per day)
        
        Returns:
            Tensor of shape (batch_size, 1024)
        """
        batch_size, num_days, context_len = X_tweet_tkids.size()
        
        # Flatten the batch for embedding lookup
        X_flat = X_tweet_tkids.view(-1, context_len)  # Shape: (batch_size * num_days, context_len)
        
        # Get embeddings from the pre-trained model
        with torch.no_grad():  # Prevent gradients through pre-trained model
            embeddings = self.embedding_model.get_input_embeddings()(X_flat)  # Shape: (batch_size * num_days, context_len, emb_dim)
        
        # Reshape for CNN
        embeddings = embeddings.view(batch_size, num_days * context_len, self.emb_dim)  # Shape: (batch_size, 30 * 40, emb_dim)
        embeddings = embeddings.permute(0, 2, 1)  # Shape: (batch_size, emb_dim, 30 * 40) for Conv1d
        
        # Apply CNN
        x = self.cnn(embeddings)  # Shape: (batch_size, 1024, 30 * 40)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Reduce along the time dimension (30 * 40) to get the final output
        x = torch.mean(x, dim=2)  # Shape: (batch_size, 1024)
        
        return x