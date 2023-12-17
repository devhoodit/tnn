# Toy Neural Network (TNN)
TNN is toy project that provide useful methods to make Neural Network with pure python  

# How to use
### Make tensor
```python
import tnn
t = tnn.Tensor(2)
```

### Simple Linear Regression
```python
import tnn
from tnn.loss_fn import MSE
x = tnn.Tensor(1)
y = tnn.Tensor(2)

a = tnn.Tensor(0, require_grad=True)
b = tnn.Tensor(1, require_grad=True)

for i in range(100):
    loss = MSE(a * x + b, y)
    print(f"{i}th iteration, loss: {loss.data}")
    loss.grad_ctx.backward(loss.data * -0.01)
    loss.grad_ctx.update()
    loss.grad_ctx.zero_grad()
```