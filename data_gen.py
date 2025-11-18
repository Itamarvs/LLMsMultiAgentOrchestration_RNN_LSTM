import numpy as np
import pandas as pd
import os

def generate_signals(n_samples=10000, seed=1):
    """
    Generates the mixed signal S and the targets.
    Noise (Amplitude and Phase) is randomized at EVERY sample t.
    """
    np.random.seed(seed)
    time = np.linspace(0, 10, n_samples, endpoint=False)
    frequencies = [1, 3, 5, 7]
    
    # Initialize pure targets for each frequency
    targets = {f: np.zeros(n_samples) for f in frequencies}
    
    # S(t) accumulation
    components_sum = np.zeros(n_samples)
    
    for f in frequencies:
        # Critical: Noise parameters change at every single sample t
        A_t = np.random.uniform(0.8, 1.2, n_samples)
        phi_t = np.random.uniform(0, 2*np.pi, n_samples)
        
        # Noisy component calculation
        noisy_component = A_t * np.sin(2 * np.pi * f * time + phi_t)
        components_sum += noisy_component
        
        # Pure target (Ground Truth)
        targets[f] = np.sin(2 * np.pi * f * time)
        
    S = components_sum / 4.0
    return time, S, targets

def create_dataset_dataframe(time, S, targets):
    """
    Stitches the data into a format suitable for sequential training.
    Order: 10k samples of Freq 1, then 10k of Freq 2, etc.
    """
    frequencies = [1, 3, 5, 7]
    dfs = []
    
    for i, f in enumerate(frequencies):
        df = pd.DataFrame()
        df['time'] = time
        df['S_t'] = S
        
        # One-hot encoding for Control input C
        # C1..C4 correspond to frequencies 1, 3, 5, 7
        c_vector = np.zeros(4, dtype=int)
        c_vector[i] = 1
        df['C1'] = c_vector[0]
        df['C2'] = c_vector[1]
        df['C3'] = c_vector[2]
        df['C4'] = c_vector[3]
        
        # The target is the pure sine wave of the CURRENT selected frequency
        df['target'] = targets[f]
        dfs.append(df)
        
    # Concatenate all 4 sequences (Total 40,000 rows)
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df

if __name__ == "__main__":
    # Create data directory
    os.makedirs("data", exist_ok=True)

    # [cite_start]1. Generate Train Data (Seed 1) [cite: 127]
    print("Generating Training Set (Seed 1)...")
    time_train, S_train, targets_train = generate_signals(n_samples=10000, seed=1)
    df_train = create_dataset_dataframe(time_train, S_train, targets_train)
    df_train.to_csv('data/train_dataset.csv', index=False)

    # [cite_start]2. Generate Test Data (Seed 11) [cite: 128]
    print("Generating Test Set (Seed 11)...")
    time_test, S_test, targets_test = generate_signals(n_samples=10000, seed=11)
    df_test = create_dataset_dataframe(time_test, S_test, targets_test)
    df_test.to_csv('data/test_dataset.csv', index=False)

    print("Done. Data saved to 'data/' folder.")
