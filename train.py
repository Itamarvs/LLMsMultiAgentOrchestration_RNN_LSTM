import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import LSTM_Filter
import os

# --- Robust Hyperparameters ---
HIDDEN_SIZE = 128
LEARNING_RATE = 0.005 
TBPTT_STEPS = 200     
EPOCHS = 50           
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_batched(filepath):
    """
    Loads data and reshapes it into (Seq_Len, Batch_Size, Features).
    We split the 40,000 rows into 4 parallel streams of 10,000 (one per freq).
    """
    df = pd.read_csv(filepath)
    X = df[['S_t', 'C1', 'C2', 'C3', 'C4']].values
    y = df['target'].values
    
    # Reshape: Total 40,000 -> 4 chunks of 10,000
    # New Shape: (10000, 4, 5)
    n_chunks = 4
    chunk_size = len(X) // n_chunks
    
    X_batched = np.array([X[i*chunk_size : (i+1)*chunk_size] for i in range(n_chunks)])
    y_batched = np.array([y[i*chunk_size : (i+1)*chunk_size] for i in range(n_chunks)])
    
    # Transpose to (Seq_Len, Batch, Feats) -> (10000, 4, ...)
    X_batched = np.transpose(X_batched, (1, 0, 2))
    y_batched = np.transpose(y_batched, (1, 0))
    
    return torch.tensor(X_batched, dtype=torch.float32), torch.tensor(y_batched, dtype=torch.float32)

def train():
    print(f"Training with Parallel Batching (Size=4) on {DEVICE}")
    
    # Load Batched Data
    X_train, y_train = load_data_batched('data/train_dataset.csv')
    X_test, y_test = load_data_batched('data/test_dataset.csv')
    
    model = LSTM_Filter(input_size=5, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_test_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        # Init Hidden State for Batch Size 4
        (h, c) = model.init_hidden(batch_size=4, device=DEVICE)
        
        optimizer.zero_grad()
        loss_buffer = 0 
        
        # Iterate over the 10,000 time steps
        seq_len = X_train.shape[0]
        
        for t in range(seq_len):
            # Input shape: (4, 5) - 4 samples, one from each freq block
            input_t = X_train[t].to(DEVICE)
            target_t = y_train[t].unsqueeze(1).to(DEVICE) # Shape (4, 1)
            
            output, (h, c) = model(input_t, (h, c))
            
            loss = criterion(output, target_t)
            loss_buffer += loss
            epoch_loss += loss.item()
            
            if (t + 1) % TBPTT_STEPS == 0:
                # Normalize loss by steps
                (loss_buffer / TBPTT_STEPS).backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                h = h.detach()
                c = c.detach()
                loss_buffer = 0
            
        avg_train_loss = epoch_loss / seq_len
        
        # --- Validation ---
        model.eval()
        test_loss = 0.0
        (h_val, c_val) = model.init_hidden(batch_size=4, device=DEVICE)
        
        with torch.no_grad():
            for t in range(X_test.shape[0]):
                input_t = X_test[t].to(DEVICE)
                target_t = y_test[t].unsqueeze(1).to(DEVICE)
                
                output, (h_val, c_val) = model(input_t, (h_val, c_val))
                test_loss += criterion(output, target_t).item()

        avg_test_loss = test_loss / X_test.shape[0]
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train MSE: {avg_train_loss:.5f} | Test MSE: {avg_test_loss:.5f}")
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')

    print(f"Training Complete. Best Test MSE: {best_test_loss:.5f}")

if __name__ == "__main__":
    train()
