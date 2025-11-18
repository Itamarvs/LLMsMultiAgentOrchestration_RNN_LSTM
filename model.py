import torch
import torch.nn as nn

class LSTM_Filter(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=1):
        super(LSTM_Filter, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM Cell
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        
        # Linear mapping
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        # Tanh output to match sine wave range [-1, 1]
        self.activation = nn.Tanh()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Smart Initialization:
        The input S[t] (index 0) is pure noise. We initialize its weight to be small
        so the model starts by ignoring it.
        The Control inputs C (indices 1-4) are critical, so we leave them standard.
        """
        # Initialize all parameters using Xavier/Kaiming
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Access the input-to-hidden weights of the LSTM cell
        # PyTorch LSTMCell weights are concatenated (W_ii|W_if|W_ig|W_io)
        # Input size is 5. We want to dampen index 0 (Signal) for all gates.
        with torch.no_grad():
            # Shape: (4*hidden_size, input_size)
            W_ih = self.lstm_cell.weight_ih
            
            # Set column 0 (Noisy Signal) to near-zero
            W_ih[:, 0] *= 0.01 
            
            # Boost columns 1-4 (Control Vectors) slightly
            W_ih[:, 1:] *= 2.0
            
    def forward(self, x, hidden_state):
        h_t, c_t = self.lstm_cell(x, hidden_state)
        out = self.fc_out(h_t)
        output = self.activation(out)
        return output, (h_t, c_t)
    
    def init_hidden(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_size).to(device),
                torch.zeros(batch_size, self.hidden_size).to(device))
