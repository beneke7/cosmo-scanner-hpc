# Deep Learning from First Principles
## A Physicist's Guide to Neural Networks

*This guide teaches machine learning by building intuition from physics foundations. Every concept is explained with code, mathematics, and physical analogies.*

---

## Table of Contents

1. [Chapter 1: The Tensor - Your Fundamental Object](#chapter-1-the-tensor)
2. [Chapter 2: Automatic Differentiation - The Chain Rule Machine](#chapter-2-automatic-differentiation)
3. [Chapter 3: The Neural Network - Function Approximation](#chapter-3-the-neural-network)
4. [Chapter 4: Convolutions - Translation Invariant Operators](#chapter-4-convolutions)
5. [Chapter 5: The Training Loop - Gradient Flow](#chapter-5-the-training-loop)
6. [Chapter 6: The DataLoader - Parallel Data Pipelines](#chapter-6-the-dataloader)
7. [Chapter 7: Putting It All Together](#chapter-7-putting-it-all-together)

---

# Chapter 1: The Tensor

## 1.1 What is a Tensor?

In physics, a tensor is a geometric object that transforms in a specific way under coordinate changes. In PyTorch, a **tensor** is simpler but related: it's an n-dimensional array that carries:

1. **Data**: The actual numbers (stored on CPU or GPU memory)
2. **Shape**: The dimensions (e.g., `[64, 64]` for a 2D field)
3. **Dtype**: The data type (`float32`, `float64`, etc.)
4. **Device**: Where it lives (`cpu`, `cuda`, `mps`)
5. **Gradient tracking**: Whether to record operations for differentiation

```python
import torch

# Creating tensors
# ================

# From Python lists
x = torch.tensor([1.0, 2.0, 3.0])
print(f"1D tensor: {x}, shape: {x.shape}")  # shape: torch.Size([3])

# Zeros and ones (like numpy)
zeros = torch.zeros(64, 64)      # 64x64 matrix of zeros
ones = torch.ones(3, 3)          # 3x3 matrix of ones

# Random tensors
uniform = torch.rand(10)         # Uniform [0, 1)
normal = torch.randn(10)         # Standard normal N(0, 1)

# From numpy (shares memory!)
import numpy as np
np_array = np.array([1, 2, 3], dtype=np.float32)
tensor_from_np = torch.from_numpy(np_array)  # No copy!
```

## 1.2 Tensor Operations = Linear Algebra

```python
# Element-wise operations (like field operations)
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

c = a + b           # [5, 7, 9]     - Addition
d = a * b           # [4, 10, 18]   - Element-wise multiplication
e = a ** 2          # [1, 4, 9]     - Element-wise power
f = torch.exp(a)    # [e¹, e², e³]  - Element-wise exponential

# Matrix operations
A = torch.randn(3, 4)  # 3x4 matrix
B = torch.randn(4, 5)  # 4x5 matrix
C = A @ B              # 3x5 matrix (matrix multiplication)
# Equivalent: C = torch.matmul(A, B)

# Broadcasting (like numpy)
# Smaller tensor is "broadcast" to match larger tensor's shape
x = torch.randn(64, 64)     # 2D field
bias = torch.randn(64)      # 1D vector
result = x + bias           # bias is broadcast along first dimension
```

## 1.3 The Shape System - Understanding Dimensions

This is **critical** for deep learning. Every tensor has a shape, and operations require compatible shapes.

```python
# Convention for images/fields in PyTorch:
# (Batch, Channels, Height, Width) = (B, C, H, W)

# A single grayscale image
single_image = torch.randn(1, 64, 64)     # (C, H, W) = (1, 64, 64)

# A batch of 32 grayscale images
batch = torch.randn(32, 1, 64, 64)        # (B, C, H, W) = (32, 1, 64, 64)

# Accessing dimensions
print(f"Batch size: {batch.shape[0]}")     # 32
print(f"Channels: {batch.shape[1]}")       # 1
print(f"Height: {batch.shape[2]}")         # 64
print(f"Width: {batch.shape[3]}")          # 64

# Reshaping
x = torch.randn(32, 128)                   # (32, 128)
x_reshaped = x.view(32, 128, 1, 1)         # (32, 128, 1, 1)
x_flat = x_reshaped.view(32, -1)           # (32, 128) - (-1 infers dimension)

# Adding/removing dimensions
x = torch.randn(64, 64)                    # (64, 64)
x_with_channel = x.unsqueeze(0)            # (1, 64, 64) - add dim at position 0
x_with_batch = x_with_channel.unsqueeze(0) # (1, 1, 64, 64) - add another dim
```

## 1.4 Device Management

Tensors must be on the same device for operations.

```python
# Check available devices
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Move tensors between devices
x_cpu = torch.randn(100, 100)              # Created on CPU by default
x_gpu = x_cpu.to('cuda')                   # Move to NVIDIA GPU
x_mps = x_cpu.to('mps')                    # Move to Apple Silicon GPU

# Create directly on device
x = torch.randn(100, 100, device='cuda')

# Operations require same device
# x_cpu + x_gpu  # ERROR! Cannot add tensors on different devices
```

---

# Chapter 2: Automatic Differentiation

## 2.1 The Computational Graph

This is the **magic** of PyTorch. When you perform operations on tensors with `requires_grad=True`, PyTorch builds a computational graph that tracks how outputs depend on inputs.

```python
# Enable gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Perform operations - PyTorch records them!
y = x ** 2          # y = [4, 9]
z = y.sum()         # z = 13 (scalar)

# The computational graph looks like:
#
#   x ──[square]──> y ──[sum]──> z
#
# PyTorch knows: z = sum(x²) = x₀² + x₁²

# Compute gradients via backpropagation
z.backward()

# Now x.grad contains ∂z/∂x
print(f"x.grad = {x.grad}")  # [4.0, 6.0] because ∂z/∂xᵢ = 2xᵢ
```

## 2.2 The Chain Rule in Action

```python
# More complex example: z = (x * w + b)²

x = torch.tensor([1.0, 2.0, 3.0])                    # Input (fixed)
w = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)  # Weights (learnable)
b = torch.tensor([0.1], requires_grad=True)            # Bias (learnable)

# Forward pass
h = x * w       # Element-wise: [0.5, 1.0, 1.5]
s = h.sum()     # Scalar: 3.0
y = s + b       # Scalar: 3.1
z = y ** 2      # Scalar: 9.61

# Backward pass - compute ALL gradients at once!
z.backward()

# Let's verify manually:
# z = (sum(x*w) + b)²
# ∂z/∂w = 2(sum(x*w) + b) * x = 2 * 3.1 * [1, 2, 3] = [6.2, 12.4, 18.6]
# ∂z/∂b = 2(sum(x*w) + b) * 1 = 2 * 3.1 = 6.2

print(f"w.grad = {w.grad}")  # tensor([6.2000, 12.4000, 18.6000])
print(f"b.grad = {b.grad}")  # tensor([6.2000])
```

## 2.3 Why This Matters: The Adjoint Method

Physicists use the **adjoint method** in optimal control and sensitivity analysis. Backpropagation is exactly this!

**Forward mode**: Compute $\frac{\partial \text{output}}{\partial \text{one input}}$ — efficient if few inputs, many outputs.

**Backward mode (Backprop)**: Compute $\frac{\partial \text{one output}}{\partial \text{all inputs}}$ — efficient if one output (loss), many inputs (parameters).

Since neural networks have millions of parameters but one scalar loss, backward mode wins.

```python
# The key insight: ONE backward pass gives ALL gradients!
# Cost = O(forward pass), not O(forward pass × num_parameters)

model_with_1M_params = ...
loss = model_with_1M_params(input)  # Forward: O(1M)
loss.backward()                      # Backward: O(1M), NOT O(1M²)!
```

---

# Chapter 3: The Neural Network

## 3.1 A Neuron = Weighted Sum + Nonlinearity

The simplest neural network unit:

$$y = \sigma(w^T x + b) = \sigma\left(\sum_i w_i x_i + b\right)$$

where $\sigma$ is a nonlinear **activation function**.

```python
import torch.nn as nn

# A single "neuron" taking 10 inputs, producing 1 output
class SingleNeuron(nn.Module):
    def __init__(self, input_size=10):
        super().__init__()
        # nn.Linear implements: y = Wx + b
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        h = self.linear(x)       # Linear transformation
        y = torch.sigmoid(h)     # Nonlinear activation: σ(h) = 1/(1+e^{-h})
        return y

neuron = SingleNeuron()
x = torch.randn(1, 10)  # One sample, 10 features
y = neuron(x)           # Output in (0, 1)
```

## 3.2 Layers = Many Neurons in Parallel

```python
# nn.Linear(in_features, out_features) is a LAYER of neurons
# It computes: Y = X @ W.T + b
# where W has shape (out_features, in_features)

layer = nn.Linear(10, 5)  # 10 inputs → 5 outputs (5 neurons)

print(f"Weight shape: {layer.weight.shape}")  # (5, 10) - each row is one neuron
print(f"Bias shape: {layer.bias.shape}")      # (5,) - one bias per neuron

x = torch.randn(32, 10)   # Batch of 32 samples, 10 features each
y = layer(x)              # Shape: (32, 5)
```

## 3.3 The Multi-Layer Perceptron (MLP)

Stack layers with nonlinearities between them:

```python
class MLP(nn.Module):
    """
    Multi-Layer Perceptron: the simplest deep network.
    
    Architecture:
        Input (10) → Linear → ReLU → Linear → ReLU → Linear → Output (1)
               
    Physical analogy: 
        Each layer is a linear transformation (rotation + scaling)
        followed by a nonlinear "activation" (like a threshold detector).
    """
    def __init__(self):
        super().__init__()
        
        # Define layers
        self.layer1 = nn.Linear(10, 64)    # 10 → 64 dimensions
        self.layer2 = nn.Linear(64, 32)    # 64 → 32 dimensions  
        self.layer3 = nn.Linear(32, 1)     # 32 → 1 dimension
        
        # Activation function
        self.relu = nn.ReLU()  # ReLU(x) = max(0, x)
    
    def forward(self, x):
        # Layer 1: Linear + ReLU
        x = self.layer1(x)    # (B, 10) → (B, 64)
        x = self.relu(x)      # Element-wise max(0, x)
        
        # Layer 2: Linear + ReLU
        x = self.layer2(x)    # (B, 64) → (B, 32)
        x = self.relu(x)
        
        # Layer 3: Linear only (no activation for regression)
        x = self.layer3(x)    # (B, 32) → (B, 1)
        
        return x

# Test it
model = MLP()
x = torch.randn(32, 10)   # Batch of 32
y = model(x)              # Shape: (32, 1)

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params}")  # 10*64 + 64 + 64*32 + 32 + 32*1 + 1 = 2,913
```

## 3.4 Why Nonlinearities?

**Theorem**: Without nonlinearities, any deep network is equivalent to a single linear layer!

$$f(x) = W_3(W_2(W_1 x)) = (W_3 W_2 W_1) x = W_{\text{effective}} x$$

Nonlinearities break this collapse and enable **universal function approximation**.

```python
# Common activation functions
x = torch.linspace(-3, 3, 100)

# ReLU: Rectified Linear Unit
# Physical analogy: A diode that only passes positive current
relu = torch.relu(x)  # max(0, x)

# Sigmoid: Squashes to (0, 1)  
# Physical analogy: Fermi-Dirac distribution (occupation probability)
sigmoid = torch.sigmoid(x)  # 1 / (1 + exp(-x))

# Tanh: Squashes to (-1, 1)
# Physical analogy: Magnetization curve
tanh = torch.tanh(x)

# GELU: Gaussian Error Linear Unit (used in transformers)
# Physical analogy: Smooth approximation to ReLU
gelu = torch.nn.functional.gelu(x)
```

---

# Chapter 4: Convolutions

## 4.1 The Convolution Operation

For 2D fields (images), convolution is:

$$(f * g)(x, y) = \sum_{i,j} g(i, j) \cdot f(x-i, y-j)$$

This is a **local, translation-invariant** operation.

```python
import torch.nn.functional as F

# Create a simple 2D field (think: density field)
field = torch.randn(1, 1, 8, 8)  # (B=1, C=1, H=8, W=8)

# Create a kernel (think: filter/detector)
kernel = torch.tensor([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=torch.float32).view(1, 1, 3, 3)  # (out_C, in_C, kH, kW)

# Apply convolution
output = F.conv2d(field, kernel, padding=1)
print(f"Input shape: {field.shape}")    # (1, 1, 8, 8)
print(f"Output shape: {output.shape}")  # (1, 1, 8, 8) with padding=1
```

## 4.2 Understanding Conv2d Parameters

```python
# nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

conv = nn.Conv2d(
    in_channels=1,      # Number of input channels (1 for grayscale)
    out_channels=32,    # Number of output channels (32 different filters)
    kernel_size=3,      # 3x3 kernel
    stride=1,           # Move kernel by 1 pixel each step
    padding=1           # Pad input with 1 pixel of zeros on each side
)

# What this creates:
# - 32 different 3x3 kernels (filters)
# - Each kernel slides across the input, computing dot products
# - Output has 32 channels (one per kernel)

print(f"Weight shape: {conv.weight.shape}")  # (32, 1, 3, 3)
# 32 output channels × 1 input channel × 3×3 kernel

x = torch.randn(16, 1, 64, 64)  # Batch of 16 grayscale 64x64 images
y = conv(x)
print(f"Output shape: {y.shape}")  # (16, 32, 64, 64)
```

## 4.3 The Convolution as a Green's Function

In physics, many problems have solutions of the form:

$$\phi(\mathbf{r}) = \int G(\mathbf{r} - \mathbf{r}') \rho(\mathbf{r}') d^3r'$$

The kernel $G$ is the **Green's function** — it encodes how a source at $\mathbf{r}'$ contributes to the field at $\mathbf{r}$.

**In CNNs**: The convolutional kernel is a **learnable** Green's function. Instead of specifying it analytically, we let gradient descent discover the optimal kernel for our task.

```python
# Example: Edge detection kernel (manually specified)
sobel_x = torch.tensor([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=torch.float32).view(1, 1, 3, 3)

# In a CNN, we don't specify the kernel — we LEARN it!
# The network discovers that edge-like patterns are useful features
```

## 4.4 Output Size Calculation

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}} \right\rfloor + 1$$

```python
def conv_output_size(H_in, kernel_size, stride=1, padding=0):
    return (H_in + 2 * padding - kernel_size) // stride + 1

# Examples:
print(conv_output_size(64, kernel_size=3, padding=0))  # 62
print(conv_output_size(64, kernel_size=3, padding=1))  # 64 (size preserved!)
print(conv_output_size(64, kernel_size=3, stride=2, padding=1))  # 32 (halved!)
```

## 4.5 Pooling = Coarse Graining

Pooling reduces spatial dimensions by summarizing local regions.

```python
# Max pooling: Take maximum in each 2x2 block
pool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.randn(1, 1, 64, 64)
y = pool(x)
print(f"After pooling: {y.shape}")  # (1, 1, 32, 32)

# Average pooling: Take mean in each block
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Adaptive pooling: Specify OUTPUT size, not kernel size
global_pool = nn.AdaptiveAvgPool2d(output_size=1)  # Any input → (1, 1)
```

**Physical analogy**: This is **block-spin renormalization**.
- Divide lattice into blocks
- Replace each block with a single "effective" value
- Repeat → flow to IR (large scales)

---

# Chapter 5: The Training Loop

## 5.1 The Optimization Problem

Training minimizes a **loss function** $\mathcal{L}(\theta)$:

$$\theta^* = \arg\min_\theta \mathcal{L}(\theta) = \arg\min_\theta \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)$$

For regression, we typically use **Mean Squared Error**:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N (f_\theta(x_i) - y_i)^2$$

```python
# Loss functions in PyTorch
mse_loss = nn.MSELoss()            # For regression
cross_entropy = nn.CrossEntropyLoss()  # For classification

# Example
predictions = torch.tensor([0.5, 0.8, 0.3])
targets = torch.tensor([0.6, 0.7, 0.4])
loss = mse_loss(predictions, targets)
print(f"MSE Loss: {loss.item():.4f}")
```

## 5.2 Gradient Descent

The fundamental update rule:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$$

```python
# Manual gradient descent (educational only!)
def manual_gradient_descent():
    # Simple quadratic loss: L(w) = w²
    w = torch.tensor([5.0], requires_grad=True)
    lr = 0.1
    
    for step in range(20):
        loss = w ** 2
        loss.backward()       # Compute gradient: dL/dw = 2w
        
        with torch.no_grad():  # Don't track this operation!
            w -= lr * w.grad   # Update: w = w - lr * grad
        
        w.grad.zero_()         # Reset gradient for next iteration
        
        print(f"Step {step}: w = {w.item():.4f}, loss = {loss.item():.4f}")

# In practice, use optimizers!
```

## 5.3 The Complete Training Loop

```python
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    One epoch = one complete pass through the training data.
    
    THE PHYSICS:
    - Each batch gives a Monte Carlo estimate of the true gradient
    - We're performing stochastic gradient descent on the loss landscape
    - The loss is our "potential energy" — we're rolling downhill
    """
    model.train()  # Enable training mode (affects dropout, batch norm)
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # 1. Move data to device (CPU/GPU)
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 2. FORWARD PASS: Compute predictions
        #    This builds the computational graph
        predictions = model(inputs)
        
        # 3. COMPUTE LOSS: Measure how wrong we are
        #    This is the "potential energy" at current θ
        loss = criterion(predictions, targets)
        
        # 4. BACKWARD PASS: Compute gradients
        #    This traverses the graph backward, computing ∂L/∂θ
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        
        # 5. OPTIMIZER STEP: Update parameters
        #    θ_new = θ_old - lr * gradient (plus momentum, etc.)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

## 5.4 Understanding Optimizers

### Stochastic Gradient Descent (SGD)

$$\theta_{t+1} = \theta_t - \eta g_t$$

where $g_t = \nabla_\theta \mathcal{L}_{\text{batch}}$ is the gradient on the current mini-batch.

### SGD with Momentum

$$v_{t+1} = \mu v_t + g_t$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

**Physical analogy**: A ball rolling down a hill with inertia. It accumulates velocity and can roll through small local minima.

### Adam (Adaptive Moment Estimation)

Keeps running averages of gradient (momentum) AND squared gradient (for adaptive learning rates):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(first moment)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(second moment)}$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

```python
# PyTorch optimizers
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer_momentum = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer_adamw = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### Weight Decay (L2 Regularization)

Adds $\frac{\lambda}{2}||\theta||^2$ to the loss:

$$\mathcal{L}_{\text{regularized}} = \mathcal{L} + \frac{\lambda}{2} \sum_i \theta_i^2$$

**Physical analogy**: A spring force $F = -\lambda \theta$ pulling parameters toward zero. Prevents weights from growing unboundedly.

---

# Chapter 6: The DataLoader

## 6.1 The Dataset Class

```python
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    """
    A Dataset defines HOW to get a single sample.
    
    You must implement:
    - __len__: How many samples?
    - __getitem__: Get sample at index i
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Usage
X = np.random.randn(1000, 10)
y = np.random.randn(1000, 1)
dataset = SimpleDataset(X, y)

print(f"Dataset size: {len(dataset)}")
sample_x, sample_y = dataset[0]
print(f"Sample shapes: x={sample_x.shape}, y={sample_y.shape}")
```

## 6.2 The DataLoader

```python
loader = DataLoader(
    dataset,
    batch_size=32,      # Samples per batch
    shuffle=True,       # Randomize order each epoch
    num_workers=4,      # Parallel data loading processes
    pin_memory=True,    # Faster CPU→GPU transfer
    drop_last=True      # Drop incomplete final batch
)

# Iterate through batches
for batch_x, batch_y in loader:
    print(f"Batch shapes: x={batch_x.shape}, y={batch_y.shape}")
    # batch_x: (32, 10), batch_y: (32, 1)
    break
```

## 6.3 On-the-Fly Generation

For simulation data, generate samples inside `__getitem__`:

```python
class GenerativeDataset(Dataset):
    """
    Generate data ON-THE-FLY instead of storing it.
    
    Advantages:
    - Infinite effective dataset size
    - No memory/disk limitations
    - Fresh samples each epoch (implicit augmentation)
    
    The key: __getitem__ runs in PARALLEL across num_workers processes!
    """
    def __init__(self, num_samples, param_range=(0.1, 0.5)):
        self.num_samples = num_samples
        self.param_min, self.param_max = param_range
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate fresh sample each time!
        param = np.random.uniform(self.param_min, self.param_max)
        field = self.generate_field(param)  # Your simulation
        
        return torch.from_numpy(field), torch.tensor([param])
    
    def generate_field(self, param):
        # Your physics simulation here
        return np.random.randn(64, 64).astype(np.float32)
```

---

# Chapter 7: Putting It All Together

## 7.1 The CosmoNet Architecture (Annotated)

```python
class CosmoNet(nn.Module):
    """
    A CNN for cosmological parameter estimation.
    
    ARCHITECTURE OVERVIEW:
    ======================
    
    Input: (B, 1, 64, 64) - Batch of density fields
           B = batch size (how many fields at once)
           1 = channels (grayscale)
           64, 64 = spatial dimensions
    
    Stage 1: FEATURE EXTRACTION (Convolutional Blocks)
    --------------------------------------------------
    Each block: Conv2d → ReLU → MaxPool
    
    Purpose: Learn hierarchical features
    - Block 1: Low-level features (edges, gradients)
    - Block 2: Mid-level features (textures, patterns)
    - Block 3: High-level features (structures, correlations)
    
    Stage 2: AGGREGATION (Global Average Pooling)
    ---------------------------------------------
    Collapse spatial dimensions to a single vector.
    Makes the network translation-invariant.
    
    Stage 3: PREDICTION (Linear Head)
    ---------------------------------
    Project from feature space to parameter space.
    """
    
    def __init__(self, input_size=64):
        super().__init__()
        
        # ===== BLOCK 1: Initial Feature Extraction =====
        # Input: (B, 1, 64, 64)
        self.conv1 = nn.Conv2d(
            in_channels=1,      # Grayscale input
            out_channels=32,    # Learn 32 different filters
            kernel_size=3,      # 3×3 local receptive field
            padding=1           # Preserve spatial size
        )
        # After conv1: (B, 32, 64, 64)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool1: (B, 32, 32, 32) — spatial size halved
        
        # ===== BLOCK 2: Intermediate Features =====
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # After conv2: (B, 64, 32, 32)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool2: (B, 64, 16, 16)
        
        # ===== BLOCK 3: High-Level Features =====
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # After conv3: (B, 128, 16, 16)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool3: (B, 128, 8, 8)
        
        # ===== GLOBAL POOLING: Spatial → Vector =====
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # After global_pool: (B, 128, 1, 1)
        # This averages over all spatial positions
        
        # ===== LINEAR HEAD: Feature → Parameter =====
        self.fc = nn.Linear(128, 1)
        # After fc: (B, 1) — the predicted Ω_m
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass: field → parameter estimate
        
        Think of this as a measurement process:
        - Input: Raw field configuration
        - Conv blocks: Extract relevant features
        - Global pool: Summarize into a single vector
        - Linear: Project to the parameter of interest
        """
        # Input validation
        assert x.dim() == 4, f"Expected 4D (B,C,H,W), got {x.dim()}D"
        assert x.size(1) == 1, f"Expected 1 channel, got {x.size(1)}"
        
        # Block 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        # Global pooling → flatten → linear
        x = self.global_pool(x)      # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)    # (B, 128)
        x = self.fc(x)               # (B, 1)
        
        return x
```

## 7.2 Complete Training Script (Annotated)

```python
def train(epochs=50, batch_size=32, lr=1e-3):
    """
    Complete training procedure with annotations.
    """
    # ===== SETUP =====
    
    # 1. Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 2. Create model and move to device
    model = CosmoNet().to(device)
    
    # 3. Loss function: MSE for regression
    # L = (1/B) Σ (pred - target)²
    criterion = nn.MSELoss()
    
    # 4. Optimizer: AdamW with weight decay
    # - Adam: Momentum + adaptive learning rates
    # - Weight decay: L2 regularization (spring force on weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    
    # 5. Learning rate scheduler: Cosine annealing
    # Slowly reduce lr following a cosine curve
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    # 6. Data loaders
    train_loader, val_loader = create_dataloaders(
        train_samples=10000,
        val_samples=1000,
        batch_size=batch_size,
        num_workers=4
    )
    
    # ===== TRAINING LOOP =====
    
    for epoch in range(1, epochs + 1):
        # --- Training phase ---
        model.train()  # Enable training mode
        train_loss = 0
        
        for fields, targets in train_loader:
            # Move data to device
            fields = fields.to(device)
            targets = targets.to(device)
            
            # Forward pass
            predictions = model(fields)
            loss = criterion(predictions, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # --- Validation phase ---
        model.eval()  # Disable training mode
        val_loss = 0
        
        with torch.no_grad():  # Don't compute gradients
            for fields, targets in val_loader:
                fields = fields.to(device)
                targets = targets.to(device)
                predictions = model(fields)
                val_loss += criterion(predictions, targets).item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Log progress
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}")
    
    return model
```

---

## Quick Reference: PyTorch Cheatsheet

### Tensor Creation
```python
torch.zeros(3, 4)          # 3×4 zeros
torch.ones(3, 4)           # 3×4 ones
torch.randn(3, 4)          # 3×4 standard normal
torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)    # [0, 0.25, 0.5, 0.75, 1]
```

### Tensor Operations
```python
x.shape, x.dtype, x.device  # Properties
x.view(2, -1)               # Reshape (-1 = infer)
x.unsqueeze(0)              # Add dimension
x.squeeze()                 # Remove dimensions of size 1
x.permute(2, 0, 1)          # Reorder dimensions
x.to('cuda')                # Move to GPU
```

### Common Layers
```python
nn.Linear(in, out)          # Fully connected
nn.Conv2d(in_ch, out_ch, k) # 2D convolution
nn.MaxPool2d(k)             # Max pooling
nn.BatchNorm2d(ch)          # Batch normalization
nn.Dropout(p)               # Dropout
nn.ReLU()                   # ReLU activation
```

### Training Pattern
```python
model.train()               # Training mode
model.eval()                # Evaluation mode
optimizer.zero_grad()       # Clear gradients
loss.backward()             # Compute gradients
optimizer.step()            # Update parameters
```

---

## Exercises

1. **Modify the architecture**: Add batch normalization after each conv layer. How does training change?

2. **Experiment with optimizers**: Compare SGD, SGD+momentum, Adam, and AdamW. Plot the loss curves.

3. **Visualize feature maps**: Use the `get_feature_maps()` method to see what each layer "sees".

4. **Add data augmentation**: Implement random rotations and flips. Does this improve generalization?

5. **Implement early stopping**: Stop training when validation loss stops improving.

---

*"What I cannot create, I do not understand."* — Richard Feynman
