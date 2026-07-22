# 🧠 mygrad

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**mygrad** is a tiny, scalar-valued autograd engine and neural network library built entirely from scratch in pure Python. It implements backpropagation dynamically over a DAG (Directed Acyclic Graph) and features a built-in Neural Network API to easily construct and train Multi-Layer Perceptrons (MLPs).

> **A massive shoutout to Andrej Karpathy (the absolute GOAT 🐐)**. This project was heavily inspired by his incredible `micrograd` library and his zero-to-hero educational series. If you want to understand how deep learning works under the hood, go watch his videos!

---

## ✨ Features

* **Autograd Engine**: A robust `Value` object that keeps track of mathematical operations (`+`, `-`, `*`, `/`, `**`, `tanh`, `exp`) and automatically computes gradients via the chain rule.
* **Neural Network API**: Clean, PyTorch-like abstractions including `Neuron`, `Layer`, and `MLP` classes to build neural networks of any size.
* **Graph Visualization**: Built-in integration with `graphviz` to render the microscopic computational graph (every single mathematical operation) and the macroscopic MLP architecture.
* **Smart Edge Rendering**: When visualizing the MLP, weights are color-coded (🔵 **Blue** = Positive, 🔴 **Red** = Negative) and line thicknesses dynamically adjust based on the magnitude of the weight! You can literally see what your network has learned.

---

## 🚀 Quick Start

### Installation
Since this is a lightweight educational tool, simply clone the repository and make sure you have `graphviz` installed to generate the visual graphs.

```bash
git clone https://github.com/yourusername/mygrad.git
cd mygrad
pip install graphviz
```

### Training a Neural Network
Here is an example of creating an MLP, feeding it a tiny dataset, and running a Gradient Descent training loop to teach it the targets.

```python
from mygrad.nn import MLP
from mygrad.engine import Value
from mygrad.visualize import draw_mlp

# 1. Initialize a model (3 inputs, two hidden layers of 4, 1 output)
model = MLP(3, [4, 4, 1])

# 2. Define a dataset
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

# 3. Training Loop (Gradient Descent)
epochs = 50
learning_rate = 0.05

for k in range(epochs):
    # Forward pass
    ypred = [model(x) for x in xs]
    
    # Calculate Mean Squared Error loss
    loss = sum(((Value(y) - yp)**2 for yp, y in zip(ypred, ys)), Value(0.0))
    
    # Zero gradients
    for p in model.parameters():
        p.grad = 0.0
        
    # Backward pass
    loss.backward()
    
    # Update weights
    for p in model.parameters():
        p.data -= learning_rate * p.grad
        
    print(f"Epoch {k+1} | Loss: {loss.data:.4f}")

# 4. Visualize the trained network!
dot = draw_mlp(model, nin=3)
dot.render("trained_network", format="svg", cleanup=True)
```

## 📊 Visualizations

![mlp after learning.svg](mlp%20after%20learning.svg)
![mlp abstract visualiztion.svg](mlp%20visualization.svg)

---

### License
MIT
