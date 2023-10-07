def null_fn(): 
    pass

class Value:
    def __init__(self, data, parent=(), op=''):
        self.data = data
        self.grad = 0
        self._backward_fn = null_fn
        self._pre = parent
        self._op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other), op='+')
        
        def _backward_add_fn():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward_fn = _backward_add_fn
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data * other.data, (self, other), op='*')

        def _backward_mul_fn():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        
        out._backward_fn = _backward_mul_fn
        return out

    def __sub__(self, other): # self - other
        return self + (-other)


    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, \
        backward={self._backward_fn.__name__})"    

    def backward(self):
        pass

if __name__ == "__main__":
    a = Value(1)
    print(a)