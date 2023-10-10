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
        return f"Value(data={self.data}, grad={self.grad})"    

    def backward(self):
        pass

    def _back_workflow(self):
        pass

    def relu(self):
        _relu = lambda d: 0 if d < 0 else d 
        out = Value(_relu(self.data), (self,), 'ReLU')
        
        def _backward_relu_fn():
            self.grad += (self.data > 0) * out.grad

        out._backward_fn = _backward_relu_fn
        return out

if __name__ == "__main__":
    b = Value(2)
    a = Value(-1).relu()
    c = a + b
    print(a, b, c)