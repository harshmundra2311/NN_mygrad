import random
from sklearn.datasets import make_moons
from mygrad.nn import MLP
from mygrad.engine import Value
import numpy as np
import matplotlib.pyplot as plt

x, y = make_moons(n_samples=100, noise=0.1)

y = y * 2 - 1

model = MLP(2, [16,16,1])

epoch = 500
learning_rate = 0.05

for k in range(epoch):
    y_pred = [model(xi) for xi in x]
    loss = sum(((Value(yi)-y_p)**2 for yi, y_p in zip(y, y_pred)), Value(0.0))/len(x)

    for p in model.parameters():
        p.grad  = 0.0
    loss.backward()

    for p in model.parameters():
        p.data -= learning_rate *  p.grad

    if((k+1)%10 == 0):
        print(f"Epoch={k+1}, Loss= {loss.data}") 
 
print("Drawing Boundary")
h = 0.25
x_min, x_max = x[:,0].min()-1, x[:,0].max()+1
y_min, y_max = x[:,1].min()-1, x[:,1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max,h), np.arange(y_min, y_max, h))

Xmesh = np.c_[xx.ravel(), yy.ravel()]

input = [list(x) for x in Xmesh]
y_pred = [model(x) for x in input]

Z = np.array([s.data > 0 for s in y_pred])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.savefig("moons_decision_boundary.png")
print("Saved to moons_decision_boundary.png!")

model.save_weights('moons_model.json')

