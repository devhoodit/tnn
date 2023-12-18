from __future__ import annotations
from typing import List
import math

# from enum import Enum, auto


# class DType(Enum):
#     Int = auto()
#     Float = auto()

# class TData():
#     def __init__(self, data, dtype=None) -> None:
#         self._data: List = []
#         if isinstance(data, (float, int)):
#             self._data.append(data)
#             self.shape = tuple()
#             if dtype == None:
#                 if isinstance(data, float):
#                     dtype = DType.Float
#                 elif isinstance(data, int):
#                     dtype = DType.Int
#             self.dtype = dtype
#             return
#         elif isinstance(data, list):
#             for d in data:
#                 self._data.append(TData(d))
#             self.shape = (len(data),) + tuple(*self._data[-1].shape)
#             self.dtype = self._data[-1].dtype
#             return
#         elif isinstance(data, TData):
#             self._data = data._data
#             self.shape = data.shape
#             self.dtype = data.dtype
#             return
#         raise ValueError()
    
#     @property
#     def data(self) -> float | int | List[TData]:
#         if self.shape == tuple():
#             return self._data[-1]
#         return self._data

#     def __add__(self, other: TData):
#         if self.dtype == other.dtype:
#             if self.shape != tuple():
#                 # array
#                 if self.shape != other.shape:
#                     raise ValueError()
#                 return TData([d1 + d2 for d1, d2 in zip(self._data, other._data)], dtype=self.dtype)
#             return TData(self._data[0] + other._data[0])
#         if self.shape == tuple():
#             # single + array
#             return TData([self.data + d1 for d1 in other._data], )
#         else:
#             # array + single
#             return TData([d1 + self.data for d1 in self._data])
#         raise ValueError()
    
#     def __sub__(self, other: TData):
#         if self.dtype == other.dtype:
#             if self.shape != tuple():
#                 # array
#                 if self.shape != other.shape:
#                     raise ValueError()
#                 return TData([d1 - d2 for d1, d2 in zip(self._data, other._data)], dtype=self.dtype)
#             return TData(self._data[0] + other._data[0])
#         if self.shape == tuple():
#             # single + array
#             return TData([self.data - d1 for d1 in other._data], )
#         else:
#             # array + single
#             return TData([d1 - self.data for d1 in self._data])
#         raise ValueError()
    
#     def __mul__(self, other: TData):
#         if self.shape == other.shape:
#             if self.shape != tuple():
#                 # array, array
#                 if self.shape != other.shape:
#                     raise ValueError()
#                 return TData([d1 * d2 for d1, d2 in zip(self._data, other._data)], dtype=self.dtype)
#             # single, single
#             return TData(self._data[0] * other._data[0])
#         if self.shape == tuple():
#             # single + array
#             return TData([self.data * d1 for d1 in other._data], )
#         else:
#             return TData([d1 * other._data[0] for d1 in self._data])
#         raise ValueError()
    
#     def __pow__(self, other: TData):
#         if self.dtype == other.dtype:
#             if self.shape != tuple():
#                 # array
#                 if self.shape != other.shape:
#                     raise ValueError()
#                 return TData([d1 ** d2 for d1, d2 in zip(self._data, other._data)], dtype=self.dtype)
#             return TData(self._data + other._data)
#         if self.shape == tuple():
#             # single + array
#             return TData([self.data ** d1 for d1 in other._data], )
#         else:
#             # array + single
#             return TData([d1 ** self.data for d1 in self._data])
#         raise ValueError()
    
# #     def __str__(self) -> str:
# #         return f"""data: {self._data}
# # shape: {self.shape}
# # type: {self.dtype}"""
    
#     def __repr__(self) -> str:
#         if self.shape == tuple():
#             return f"{self._data[0]}"
#         return f"{self._data}"

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
            self.next_ctx[1].backward(grad * self.t1.data ** (self.t2.data) * math.log(self.t1.data))
