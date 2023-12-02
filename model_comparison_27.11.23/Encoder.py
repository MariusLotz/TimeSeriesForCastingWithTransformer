import torch
import torch.nn as nn

# Define a Transformer Encoder block with MultiHead Attention
class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=1, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(input_size, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(input_size)
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # Permute the input for MultiHead Attention (from shape: sequence_length, batch_size, input_size)
        x = x.permute(1, 0, 2)  # Now: batch_size, sequence_length, input_size

        # MultiHead Attention
        attn_output, _ = self.attention(x, x, x)  # Using self-attention
        attn_output = self.layer_norm(attn_output + x)

        # Predict a single time point
        output = self.linear(attn_output[:, -1, :])  # Predict the last time point
        return output

# Example usage
input_size = 1  # Assuming univariate time series data
hidden_size = 64
num_heads = 1
dropout = 0.2

# Create an instance of the Transformer Encoder Block
model = TransformerEncoderBlock(input_size, hidden_size, num_heads, dropout)

batch_size = 32
sequence_length = 10
sample_input = torch.randn(batch_size, sequence_length, input_size)
sample_target = torch.randn(batch_size, 1)  # Example target for demonstration purposes

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(sample_input)  # Forward pass
    output = output[:, 0]
    
    # Calculate loss
    loss = criterion(output, sample_target)
    
    # Backpropagation
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Sample validation (using the same training data for illustration purposes)
model.eval()
with torch.no_grad():
    val_output = model(sample_input)
    val_loss = criterion(val_output, sample_target)
    print(f'Validation Loss: {val_loss.item()}')

