from mygrad.nn import MLP
from mygrad.engine import Value
from mygrad.visualize import draw_dot, draw_mlp
xs = [
    [0,0],
    [0,1],
    [1,0],
    [1,1],
]
ys = [-1,1,1,-1]

model = MLP(2, [8,1])

epoch = 5000
learning_rate = 0.05

for i in range(epoch):
    y_pred = [model(x) for x in xs]
    loss = sum(((Value(y) - y_pred)**2 for y, y_pred in zip(ys, y_pred)), Value(0.0))
    for p in model.parameters():
        p.grad = 0.0
    
    loss.backward()
    
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if i%100 == 0:
        print("Epoch = ", i, "Loss = ", loss)

    if i==epoch-1:
        dot = draw_dot(loss)
        dot.render("xor mlp after learning", format = "svg", cleanup = True)
        draw_mlp(model, 2).render("xor mlp visualization", format="svg", cleanup = True)

print("Ypred =", [model(x).data for x in xs])
