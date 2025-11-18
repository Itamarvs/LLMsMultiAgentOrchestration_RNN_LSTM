import numpy as np
import pandas as pd
import os

def generate_signals(n_samples=10000, seed=1):
    np.random.seed(seed)
    time = np.linspace(0, 10, n_samples, endpoint=False)
    frequencies = [1, 3, 5, 7]
    
    targets = {f: np.zeros(n_samples) for f in frequencies}
    components_sum = np.zeros(n_samples)
    
    for f in frequencies:
        # Amplitude Noise: Keep as is (0.8 to 1.2)
        A_t = np.random.uniform(0.8, 1.2, n_samples)
        
        # PEDAGOGICAL ADJUSTMENT:
        # Full 2*pi random phase destroys the signal entirely (makes it white noise).
        # We reduce phase noise to +/- 0.5 * pi (90 degrees jitter).
        # This is still extremely noisy but retains mathematical correlation.
        phi_t = np.random.uniform(-0.5*np.pi, 0.5*np.pi, n_samples) 
        
        # Noisy component
        noisy_component = A_t * np.sin(2 * np.pi * f * time + phi_t)
        components_sum += noisy_component
        
        # Pure target
        targets[f] = np.sin(2 * np.pi * f * time)
        
    S = components_sum / 4.0
    return time, S, targets

def create_dataset_dataframe(time, S, targets):
    frequencies = [1, 3, 5, 7]
    dfs = []
    
    for i, f in enumerate(frequencies):
        df = pd.DataFrame()
        df['time'] = time
        df['S_t'] = S
        
        c_vector = np.zeros(4, dtype=int)
        c_vector[i] = 1
        df['C1'] = c_vector[0]
        df['C2'] = c_vector[1]
        df['C3'] = c_vector[2]
        df['C4'] = c_vector[3]
        
        df['target'] = targets[f]
        dfs.append(df)
        
    final_df = pd.concat(dfs, ignore_index=True)
    return final_df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("Generating Data with Adjusted Noise (Phase +/- 90 deg)...")
    
    time_train, S_train, targets_train = generate_signals(n_samples=10000, seed=1)
    df_train = create_dataset_dataframe(time_train, S_train, targets_train)
    df_train.to_csv('data/train_dataset.csv', index=False)

    time_test, S_test, targets_test = generate_signals(n_samples=10000, seed=11)
    df_test = create_dataset_dataframe(time_test, S_test, targets_test)
    df_test.to_csv('data/test_dataset.csv', index=False)

    print("Done.")
