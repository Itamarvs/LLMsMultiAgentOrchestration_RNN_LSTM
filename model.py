import torch
import torch.nn as nn

class LSTM_Filter(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=1):
        """
        LSTM Filter Model.
        input_size: 5 (1 signal + 4 one-hot encoding)
        hidden_size: 128 (Internal filters)
        """
        super(LSTM_Filter, self).__init__()
        self.hidden_size = hidden_size
        
        # Using LSTMCell to manually handle state (h_t, c_t) at every step
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        
        # Linear layer to map hidden state to single scalar output
        self.fc_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden_state):
        """
        Forward pass for a single time step (L=1).
        x: Input vector [S[t], C1, C2, C3, C4]
        hidden_state: Tuple (h_prev, c_prev)
        """
        # Update internal state
        h_t, c_t = self.lstm_cell(x, hidden_state)
        
        # Predict output based on new hidden state
        output = self.fc_out(h_t)
        
        return output, (h_t, c_t)
    
    def init_hidden(self, batch_size, device):
        """Initializes hidden state and cell state with zeros."""
        return (torch.zeros(batch_size, self.hidden_size).to(device),
                torch.zeros(batch_size, self.hidden_size).to(device))
