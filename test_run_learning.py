from mygrad.nn import MLP
from mygrad.engine import Value
from mygrad.visualize import draw_dot, draw_mlp
model = MLP(3, [4,4,1])
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

epoch = 50
learning_rate = 0.05

for k in range(epoch):

    ypred = [model(x) for x in xs]

    loss = sum(((Value(y)-yp)**2 for yp, y in zip(ypred, ys)), Value(0.0))

    for p in model.parameters():
        p.grad = 0.0

    loss.backward()

    for p in model.parameters():
        p.data -= learning_rate * p.grad

    print("Epoch= ", k+1, "loss= ", loss)
dot = draw_dot(loss)
dot.render("mlp after learning", format = "svg", cleanup = True)
draw_mlp(model, 3).render("mlp visualization", format="svg", cleanup = True)
print("Ypred = ", [model(x).data for x in xs])