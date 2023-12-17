from tnn.tensor import Tensor

def MSE(t1: Tensor, t2: Tensor) -> Tensor:
    return (t1 - t2) ** 2
