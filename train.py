import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import LSTM_Filter
import os

# --- Hyperparameters Adjusted for Convergence ---
HIDDEN_SIZE = 128
# Increased LR to help escape the local minimum of predicting "0"
LEARNING_RATE = 0.005 
# Increased TBPTT steps to 500.
# At 1000Hz sampling, 500 steps = 0.5 seconds.
# This allows the network to see half a cycle of the 1Hz wave,
# which is enough to learn the curvature/oscillation dynamics.
TBPTT_STEPS = 500     
EPOCHS = 100 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df[['S_t', 'C1', 'C2', 'C3', 'C4']].values
    y = df['target'].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train():
    print(f"Training on {DEVICE} with Truncated BPTT (Steps={TBPTT_STEPS})")
    
    X_train, y_train = load_data('data/train_dataset.csv')
    X_test, y_test = load_data('data/test_dataset.csv')
    
    model = LSTM_Filter(input_size=5, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_test_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        # Initialize Hidden State
        (h, c) = model.init_hidden(batch_size=1, device=DEVICE)
        
        optimizer.zero_grad()
        loss_buffer = 0 
        
        # Iterate through the continuous sequence
        for t in range(len(X_train)):
            # Input L=1
            input_t = X_train[t].unsqueeze(0).to(DEVICE)
            target_t = y_train[t].unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # Forward pass (maintaining connection to past)
            output, (h, c) = model(input_t, (h, c))
            
            loss = criterion(output, target_t)
            loss_buffer += loss
            epoch_loss += loss.item()
            
            # Truncated Backprop: Update weights every TBPTT_STEPS
            if (t + 1) % TBPTT_STEPS == 0:
                loss_buffer.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Detach state to cut graph, but keep values for next segment
                h = h.detach()
                c = c.detach()
                loss_buffer = 0

            # Reset state at frequency block boundaries (10k, 20k, 30k)
            if (t + 1) % 10000 == 0:
                (h, c) = model.init_hidden(batch_size=1, device=DEVICE)
                loss_buffer = 0
                optimizer.zero_grad()
            
        avg_train_loss = epoch_loss / len(X_train)
        
        # --- Validation ---
        model.eval()
        test_loss = 0.0
        (h_val, c_val) = model.init_hidden(batch_size=1, device=DEVICE)
        
        with torch.no_grad():
            for t in range(len(X_test)):
                input_t = X_test[t].unsqueeze(0).to(DEVICE)
                target_t = y_test[t].unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                output, (h_val, c_val) = model(input_t, (h_val, c_val))
                test_loss += criterion(output, target_t).item()
                
                if (t + 1) % 10000 == 0:
                    (h_val, c_val) = model.init_hidden(batch_size=1, device=DEVICE)

        avg_test_loss = test_loss / len(X_test)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train MSE: {avg_train_loss:.5f} | Test MSE: {avg_test_loss:.5f}")
        
        # Save Best Model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')
            # print("  -> Best Model Saved!") # Optional: reduce clutter

    print(f"Training Complete. Best Test MSE: {best_test_loss:.5f}")

if __name__ == "__main__":
    train()
