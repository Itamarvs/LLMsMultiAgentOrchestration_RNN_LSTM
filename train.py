import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import LSTM_Filter
import os

# Hyperparameters
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 50  # Enough to converge for this task
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(filepath):
    df = pd.read_csv(filepath)
    # Extract inputs: S_t, C1, C2, C3, C4
    X = df[['S_t', 'C1', 'C2', 'C3', 'C4']].values
    # Extract target
    y = df['target'].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train():
    print(f"Training on {DEVICE}")
    
    # Load Data
    X_train, y_train = load_data('data/train_dataset.csv')
    X_test, y_test = load_data('data/test_dataset.csv')
    
    # Initialize Model
    model = LSTM_Filter(input_size=5, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_test_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # Initialize Hidden State (h, c) for the start of the epoch
        # Because our data is one long continuous sequence (per freq), 
        # we maintain state across the loop but detach gradients.
        (h, c) = model.init_hidden(batch_size=1, device=DEVICE)
        
        for t in range(len(X_train)):
            # 1. Prepare Input for this single time step (L=1)
            # Reshape to (batch_size=1, input_size=5)
            input_t = X_train[t].unsqueeze(0).to(DEVICE) 
            target_t = y_train[t].unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # 2. Forward Pass
            # We pass the previous state (h, c)
            output, (h, c) = model(input_t, (h, c))
            
            # 3. Compute Loss
            loss = criterion(output, target_t)
            
            # 4. Backward Pass & Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 5. CRITICAL: Detach state to truncate backpropagation
            # We keep the VALUES of h and c for the next step's context,
            # but we cut the connection to the previous computation graph.
            # If we don't do this, PyTorch will try to backprop through 
            # the entire history of the dataset (10k+ steps), causing OOM.
            h = h.detach()
            c = c.detach()
            
            # Reset state if we switch frequency blocks (every 10,000 samples)
            # This helps the model adapt faster to the context switch
            if (t + 1) % 10000 == 0:
                 (h, c) = model.init_hidden(batch_size=1, device=DEVICE)
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(X_train)
        
        # --- Validation Loop ---
        model.eval()
        test_loss = 0.0
        (h_val, c_val) = model.init_hidden(batch_size=1, device=DEVICE)
        
        with torch.no_grad():
            for t in range(len(X_test)):
                input_t = X_test[t].unsqueeze(0).to(DEVICE)
                target_t = y_test[t].unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                output, (h_val, c_val) = model(input_t, (h_val, c_val))
                
                loss = criterion(output, target_t)
                test_loss += loss.item()
                
                if (t + 1) % 10000 == 0:
                    (h_val, c_val) = model.init_hidden(batch_size=1, device=DEVICE)

        avg_test_loss = test_loss / len(X_test)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
        
        # Save Best Model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')
            print("  -> Model Saved!")

if __name__ == "__main__":
    train()
