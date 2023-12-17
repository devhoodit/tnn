from __future__ import annotations
from os import terminal_size
from turtle import forward
from typing import List

class Tensor():
    def __init__(self, data: float, require_grad=False) -> None:
        self.data = data
        self.grad = 0.0
        self.is_leaf = True
        self.require_grad = require_grad
        self.grad_ctx: GradContext = AccumulateContext(self)

    def __add__(self, other: float | Tensor) -> Tensor:
        if isinstance(other, (float, int)):
            other = Tensor(other)
        addCTX = AddContext(self, other)
        return addCTX.forward()
    
    def __sub__(self, other: float | Tensor) -> Tensor:
        if isinstance(other, (float, int)):
            other = Tensor(other)
        subCTX = SubContext(self, other)
        return subCTX.forward()
    
    def __mul__(self, other: float | Tensor) -> Tensor:
        if isinstance(other, (float, int)):
            other = Tensor(other)
        mulCTX = MulContext(self, other)
        return mulCTX.forward()
    
    def __pow__(self, other: float | Tensor) -> Tensor:
        if isinstance(other, (float, int)):
            other = Tensor(other)
        powCTX = PowContext(self, other)
        return powCTX.forward()
    
    def __truediv__(self, other: float | Tensor) -> Tensor:
        if isinstance(other, (float, int)):
            other = Tensor(other)
        powCTX = PowContext(other, Tensor(-1))
        flip_t = powCTX.forward()
        mulCTX = MulContext(self, flip_t)
        return mulCTX.forward()
    
    def __str__(self) -> str:
        return f"""data:{self.data}
grad: {self.grad}"""

class GradContext():
    def __init__(self, tensors: List[Tensor] | None=None) -> None:
        self.next_ctx: List[GradContext] = []
        if tensors is not None:
            self._tensors = tensors
        else:
            self._tensors: List[Tensor] = []

    def forward(self) -> Tensor:
        raise NotImplementedError
    
    def backward(self, grad: float) -> None:
        raise NotImplementedError
    
    def update(self) -> None:
        for t in self._tensors:
            t.data -= t.grad
        for c in self.next_ctx:
            c.update()
    
    def zero_grad(self) -> None:
        for t in self._tensors:
            t.grad = 0
        for c in self.next_ctx:
            c.zero_grad()

class AccumulateContext(GradContext):
    def __init__(self, t1: Tensor) -> None:
        super().__init__([t1])
        self.t1 = t1
    
    def forward(self) -> Tensor:
        return self.t1
    
    def backward(self, grad: float) -> None:
        if not self.t1.require_grad:
            return
        self.t1.grad += grad


class AddContext(GradContext):
    def __init__(self, t1: Tensor, t2: Tensor) -> None:
        super().__init__([t1, t2])
        self.t1 = t1
        self.t2 = t2

    def forward(self) -> Tensor:
        data = Tensor(self.t1.data + self.t2.data, require_grad=True)
        self.next_ctx = [self.t1.grad_ctx, self.t2.grad_ctx]
        data.grad_ctx = self
        data.is_leaf = False
        return data
    
    def backward(self, grad: float) -> None:
        if self.t1.require_grad:
            self.next_ctx[0].backward(grad)
        if self.t2.require_grad:
            self.next_ctx[1].backward(grad)

class SubContext(GradContext):
    def __init__(self, t1: Tensor, t2: Tensor) -> None:
        super().__init__([t1, t2])
        self.t1 = t1
        self.t2 = t2

    def forward(self) -> Tensor:
        data = Tensor(self.t1.data - self.t2.data, require_grad=True)
        self.next_ctx = [self.t1.grad_ctx, self.t2.grad_ctx]
        data.grad_ctx = self
        data.is_leaf = False
        return data
    
    def backward(self, grad: float) -> None:
        if self.t1.require_grad:
            self.next_ctx[0].backward(grad)
        if self.t2.require_grad:
            self.next_ctx[1].backward(-grad)

class MulContext(GradContext):
    def __init__(self, t1: Tensor, t2: Tensor) -> None:
        super().__init__([t1, t2])
        self.t1 = t1
        self.t2 = t2
    
    def forward(self) -> Tensor:
        data = Tensor(self.t1.data * self.t2.data, require_grad=True)
        self.next_ctx = [self.t1.grad_ctx, self.t2.grad_ctx]
        data.grad_ctx = self
        data.is_leaf = False
        return data

    def backward(self, grad: float) -> None:
        if self.t1.require_grad:
            self.next_ctx[0].backward(grad * self.t2.data)
        if self.t2.require_grad:
            self.next_ctx[1].backward(grad * self.t1.data)
    
class PowContext(GradContext):
    def __init__(self, t1: Tensor, t2: Tensor) -> None:
        super().__init__([t1, t2])
        self.t1 = t1
        self.t2 = t2

    def forward (self) -> Tensor:
        data = Tensor(self.t1.data ** self.t2.data, require_grad=True)
        self.next_ctx = [self.t1.grad_ctx, self.t2.grad_ctx]
        data.grad_ctx = self
        data.is_leaf = False
        return data
    
    def backward(self, grad: float) -> None:
        if self.t1.require_grad:
            self.next_ctx[0].backward(grad * self.t2.data * (self.t1.data ** (self.t2.data-1)))
        if self.t2.require_grad:
            self.next_ctx[1].backward(grad * self.t1.data ** (self.t2.data))
