from typing import List
from h2autograd import Value
import random

class Module:
    def __init__(self):
        self.requires_grad = True

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Relu(Module):  
    def __init__(self):
        super().__init__()

    def __call__(self, x:Value):
        return x.relu()
    
    def __repr__(self):
        return f"Relu()"

#当前输出大小只能为1
class Linear(Module):
    def __init__(self, in_size, weights:List=None, out_size=1, bias=0):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.w = [Value(random.uniform(-1, 1)) for _ in range(in_size)]
        self.b = Value(bias)

    def __call__(self, x):
        return sum((wi*xi for wi, xi in zip(self.w, x)), self.b)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Linear({len(self.w)})"