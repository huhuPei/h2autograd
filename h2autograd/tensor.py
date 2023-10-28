from typing import List
from h2autograd.scalar import Value

class ShapeError(Exception):
    pass

class Tensor:
    def __init__(self, data:List):
        if isinstance(data[0], (int, Value)):
            self.values = [Value(v) if isinstance(v, int) else v for v in data]
            self.rows, self.cols = len(data), None
        else:
            self.values = [[Value(v) for v in row] for row in data]
            self.rows, self.cols = len(data), len(data[0])

    @property
    def shape(self):
        return (self.rows, self.cols) if self.cols else (self.rows, )   

    def __iter__(self):
        return iter(self.values)

    # 仅支持一维x, x需放在左边, *表示矩阵乘法
    """
    example:
        w = Tensor([[1, 2, 3], [3, 4 ,5]])
        x = Tensor([1, 2, 3])
        z = w * x
    """
    def __mul__(self, other):
        return Tensor([sum((wi*xi for wi, xi in zip(w, other))) for w in self.values])

    # 相加的条件：矩阵维度一致; 或者右操作数为一维数据, 且维度等于左操作数的第二维
    """
    example:
        a = Tensor([[1, 1, 1], [1, 1, 1]])
        b = Tensor([1, 2, 3])
        c = a + b
    """
    def __add__(self, other):
        if len(self.shape) == 2 and len(other.shape) == 1 and self.shape[1] == other.shape[0]:
            pass
        elif self.shape == other.shape:
            pass
        else:
            raise ShapeError("shape of tensors are not same or broadcast failed.")
        return 

    def __getitem__(self, key):
        return self.values[key]

    def __repr__(self):
        return f"Tensor({self.values})"