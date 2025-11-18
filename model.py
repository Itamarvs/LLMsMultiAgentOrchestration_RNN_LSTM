import torch
import torch.nn as nn

class LSTM_Filter(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=1):
        super(LSTM_Filter, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM Cell (Manual state management)
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        
        # Linear mapping
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        # FIX 1: Add Tanh to force output into [-1, 1] range (matching Sine wave)
        self.activation = nn.Tanh()
        
    def forward(self, x, hidden_state):
        h_t, c_t = self.lstm_cell(x, hidden_state)
        
        # Linear projection
        out = self.fc_out(h_t)
        
        # Apply Tanh
        output = self.activation(out)
        
        return output, (h_t, c_t)
    
    def init_hidden(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_size).to(device),
                torch.zeros(batch_size, self.hidden_size).to(device))
