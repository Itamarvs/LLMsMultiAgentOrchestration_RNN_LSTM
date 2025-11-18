import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import LSTM_Filter
import os

# --- Tuning for "Escape Velocity" ---
HIDDEN_SIZE = 128
LEARNING_RATE = 0.01  # Increased LR to force oscillation learning
TBPTT_STEPS = 200     # Reduced slightly to allow more frequent updates
EPOCHS = 20           # Should converge faster with high LR
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df[['S_t', 'C1', 'C2', 'C3', 'C4']].values
    y = df['target'].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train():
    print(f"Training on {DEVICE} | LR={LEARNING_RATE} | Steps={TBPTT_STEPS}")
    
    X_train, y_train = load_data('data/train_dataset.csv')
    X_test, y_test = load_data('data/test_dataset.csv')
    
    model = LSTM_Filter(input_size=5, hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_test_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        (h, c) = model.init_hidden(batch_size=1, device=DEVICE)
        optimizer.zero_grad()
        loss_buffer = 0 
        
        for t in range(len(X_train)):
            input_t = X_train[t].unsqueeze(0).to(DEVICE)
            target_t = y_train[t].unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            output, (h, c) = model(input_t, (h, c))
            
            loss = criterion(output, target_t)
            loss_buffer += loss
            epoch_loss += loss.item()
            
            if (t + 1) % TBPTT_STEPS == 0:
                # FIX 2: Normalize loss by steps to keep gradient scale consistent
                (loss_buffer / TBPTT_STEPS).backward()
                
                # Gentle clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                h = h.detach()
                c = c.detach()
                loss_buffer = 0

            if (t + 1) % 10000 == 0:
                (h, c) = model.init_hidden(batch_size=1, device=DEVICE)
                loss_buffer = 0
                optimizer.zero_grad()
            
        avg_train_loss = epoch_loss / len(X_train)
        
        # Validation
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
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')

    print(f"Training Complete. Best Test MSE: {best_test_loss:.5f}")

if __name__ == "__main__":
    train()
