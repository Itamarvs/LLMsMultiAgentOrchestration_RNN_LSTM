import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import LSTM_Filter
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_and_plot():
    # Load Data
    df_test = pd.read_csv('data/test_dataset.csv')
    X_test = torch.tensor(df_test[['S_t', 'C1', 'C2', 'C3', 'C4']].values, dtype=torch.float32)
    y_test = df_test['target'].values
    
    # Load Model
    model = LSTM_Filter(input_size=5, hidden_size=128).to(DEVICE)
    if os.path.exists('best_lstm_model.pth'):
        model.load_state_dict(torch.load('best_lstm_model.pth', map_location=DEVICE))
        print("Loaded best model.")
    else:
        print("Warning: No trained model found. Initializing random model.")

    model.eval()
    predictions = []
    
    # Run Inference (Preserving State)
    (h, c) = model.init_hidden(batch_size=1, device=DEVICE)
    
    with torch.no_grad():
        for t in range(len(X_test)):
            input_t = X_test[t].unsqueeze(0).to(DEVICE)
            output, (h, c) = model(input_t, (h, c))
            predictions.append(output.item())
            
            # Reset state between freq blocks for cleaner evaluation visualization
            if (t + 1) % 10000 == 0:
                (h, c) = model.init_hidden(batch_size=1, device=DEVICE)

    predictions = np.array(predictions)
    
    # --- Metrics ---
    mse = np.mean((predictions - y_test)**2)
    print(f"Final Test MSE: {mse:.6f}")
    
    # --- Visualization 1: Single Frequency Analysis (f2 = 3Hz) ---
    # Data layout: 0-10k (1Hz), 10k-20k (3Hz), 20k-30k (5Hz), 30k-40k (7Hz)
    # We focus on the 3Hz block (Indices 10,000 to 20,000)
    
    start_idx = 10000
    end_idx = 10200 # Zoom in on first 200 samples of this block
    
    time_slice = df_test['time'][start_idx:end_idx]
    noisy_input = df_test['S_t'][start_idx:end_idx]
    pure_target = df_test['target'][start_idx:end_idx]
    pred_slice = predictions[start_idx:end_idx]
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_slice, noisy_input, label='Mixed Noisy Input (S)', color='lightgray', alpha=0.7)
    plt.plot(time_slice, pure_target, label='Target Pure Sine (3Hz)', color='green', linewidth=2)
    plt.plot(time_slice, pred_slice, label='LSTM Output', color='red', linestyle='--', linewidth=2)
    plt.title('Graph 1: Frequency Extraction (f=3Hz) - Noise vs Target vs Prediction')
    plt.legend()
    plt.grid(True)
    plt.savefig('graph1_single_freq.png')
    print("Saved graph1_single_freq.png")
    
    # --- Visualization 2: All Frequencies ---
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    freqs = [1, 3, 5, 7]
    
    for i, f in enumerate(freqs):
        # Calculate range for this frequency block
        # Each block is 10,000 samples
        block_start = i * 10000
        # Visualize a window from the middle of the block to show steady state
        viz_start = block_start + 1000
        viz_end = viz_start + 100
        
        t_data = df_test['time'][viz_start:viz_end]
        target_data = df_test['target'][viz_start:viz_end]
        pred_data = predictions[viz_start:viz_end]
        
        axs[i].plot(t_data, target_data, 'g-', label='Target')
        axs[i].plot(t_data, pred_data, 'r--', label='LSTM Output')
        axs[i].set_ylabel(f'Freq {f} Hz')
        axs[i].legend(loc='upper right')
        axs[i].grid(True)
        
    axs[0].set_title('Graph 2: Extraction Performance Across All Frequencies')
    plt.tight_layout()
    plt.savefig('graph2_all_freqs.png')
    print("Saved graph2_all_freqs.png")

if __name__ == "__main__":
    evaluate_and_plot()
