# PyTorch Quick Reference Card

## Tensor Creation

```python
torch.zeros(3, 4)               # All zeros
torch.ones(3, 4)                # All ones
torch.randn(3, 4)               # Standard normal N(0,1)
torch.rand(3, 4)                # Uniform [0, 1)
torch.arange(0, 10, 2)          # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)         # [0, 0.25, 0.5, 0.75, 1]
torch.tensor([1, 2, 3])         # From list
torch.from_numpy(np_array)      # From numpy (shares memory!)
```

## Tensor Properties

```python
x.shape                         # Dimensions
x.dtype                         # Data type (float32, int64, etc.)
x.device                        # Where it lives (cpu, cuda, mps)
x.requires_grad                 # Is gradient tracking enabled?
```

## Shape Operations

```python
x.view(2, 3)                    # Reshape (must have same total elements)
x.reshape(2, -1)                # Reshape (-1 = infer dimension)
x.unsqueeze(0)                  # Add dimension at position 0
x.squeeze()                     # Remove dimensions of size 1
x.squeeze(1)                    # Remove dimension 1 if size is 1
x.permute(2, 0, 1)              # Reorder dimensions
x.transpose(0, 1)               # Swap two dimensions
x.flatten()                     # Flatten to 1D
x.view(x.size(0), -1)           # Flatten keeping batch dimension
```

## Mathematical Operations

```python
# Element-wise
x + y, x - y, x * y, x / y      # Basic arithmetic
x ** 2                          # Power
torch.exp(x)                    # Exponential
torch.log(x)                    # Natural log
torch.sqrt(x)                   # Square root
torch.abs(x)                    # Absolute value

# Reductions
x.sum()                         # Sum all elements
x.mean()                        # Mean
x.std()                         # Standard deviation
x.max(), x.min()                # Max/min value
x.argmax(), x.argmin()          # Index of max/min
x.sum(dim=0)                    # Sum along dimension 0

# Matrix operations
A @ B                           # Matrix multiplication
torch.matmul(A, B)              # Same as @
torch.mm(A, B)                  # Only for 2D matrices
A.T                             # Transpose
torch.inverse(A)                # Matrix inverse
```

## Device Management

```python
# Check availability
torch.cuda.is_available()       # CUDA (NVIDIA GPU)
torch.backends.mps.is_available()  # MPS (Apple Silicon)

# Move tensors
x.to('cuda')                    # To NVIDIA GPU
x.to('mps')                     # To Apple Silicon
x.to('cpu')                     # To CPU
x.cuda()                        # Shorthand for .to('cuda')
x.cpu()                         # Shorthand for .to('cpu')

# Create on device
torch.randn(3, 4, device='cuda')
```

## Automatic Differentiation

```python
# Enable gradient tracking
x = torch.tensor([1.0], requires_grad=True)

# Compute gradients
loss.backward()                 # Backpropagate
x.grad                          # Access gradient

# Disable gradient tracking
with torch.no_grad():
    # Operations here don't track gradients
    y = model(x)

# Detach from graph
x.detach()                      # Returns tensor without grad tracking

# Zero gradients (important!)
optimizer.zero_grad()           # Before each backward pass
x.grad.zero_()                  # Zero specific gradient
```

## Common Layers

```python
import torch.nn as nn

# Linear (fully connected)
nn.Linear(in_features, out_features)

# Convolution
nn.Conv2d(in_channels, out_channels, kernel_size)
nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

# Pooling
nn.MaxPool2d(kernel_size)
nn.AvgPool2d(kernel_size)
nn.AdaptiveAvgPool2d(output_size)  # Output size, not kernel

# Normalization
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)

# Dropout (regularization)
nn.Dropout(p=0.5)
nn.Dropout2d(p=0.5)             # For conv layers

# Activations
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.GELU()
nn.LeakyReLU(negative_slope=0.01)
```

## Building a Model

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.fc = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Define forward pass
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

model = MyModel()
```

## Loss Functions

```python
nn.MSELoss()                    # Mean Squared Error (regression)
nn.L1Loss()                     # Mean Absolute Error
nn.CrossEntropyLoss()           # Classification (includes softmax)
nn.BCELoss()                    # Binary Cross Entropy
nn.BCEWithLogitsLoss()          # BCE + sigmoid (more stable)
```

## Optimizers

```python
import torch.optim as optim

optim.SGD(model.parameters(), lr=0.01)
optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optim.Adam(model.parameters(), lr=0.001)
optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

## Learning Rate Schedulers

```python
from torch.optim.lr_scheduler import ...

StepLR(optimizer, step_size=10, gamma=0.1)    # Multiply by gamma every step_size epochs
CosineAnnealingLR(optimizer, T_max=100)       # Cosine decay
ReduceLROnPlateau(optimizer, patience=10)     # Reduce when metric plateaus
```

## Training Loop Template

```python
model.train()                   # Training mode
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Move to device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()                    # Evaluation mode
with torch.no_grad():
    predictions = model(test_x)
```

## DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = MyDataset(X, y)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

## Saving and Loading

```python
# Save model weights
torch.save(model.state_dict(), 'model.pt')

# Load model weights
model.load_state_dict(torch.load('model.pt'))

# Save everything (for resuming training)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# Load everything
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Debugging Tips

```python
# Print tensor info
print(f"Shape: {x.shape}, dtype: {x.dtype}, device: {x.device}")

# Check for NaN
torch.isnan(x).any()

# Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean={param.grad.mean():.4f}")

# Register hooks to inspect intermediate values
def print_hook(module, input, output):
    print(f"{module.__class__.__name__}: {output.shape}")

model.conv1.register_forward_hook(print_hook)
```
