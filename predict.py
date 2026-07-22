import numpy as np
import matplotlib.pyplot as plt
from mygrad.nn import MLP

# 1. Load the Brain
model = MLP(2, [16,16,1])
model.load_weights('moons_model.json')

# 2. Get User Input
print("\n--- Moon Predictor ---")
a = float(input("Enter x coordinate (e.g. 0.5): "))
b = float(input("Enter y coordinate (e.g. -0.5): "))

# 3. Make Prediction
prediction = model([a, b])
print(f"\nRaw output: {prediction.data:.4f}")

if prediction.data > 0:
    print("🔮 Prediction: BLUE MOON")
else:
    print("🔮 Prediction: RED MOON")

# 4. Visualize it!
print("Drawing prediction on the map...")

# Create the canvas
h = 0.25
x_min, x_max = -2.5, 3.5  # Hardcoded roughly to the make_moons bounds
y_min, y_max = -2.0, 2.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]

# Feed the grid to the loaded model
inputs = [list(xrow) for xrow in Xmesh]
scores = [model(xrow) for xrow in inputs]
Z = np.array([s.data > 0 for s in scores]).reshape(xx.shape)

# Draw the map
plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

# DRAW THE PREDICTION AS A GIANT YELLOW STAR
plt.scatter([a], [b], color='yellow', marker='*', s=500, edgecolor='black', zorder=5)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"Prediction for ({a}, {b})")
plt.savefig("prediction_visualized.png")
print("Saved map to prediction_visualized.png! Go check it out!")
