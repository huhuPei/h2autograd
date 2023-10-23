import collections

def null_fn(): 
    pass

class Value:
    def __init__(self, data, parents=(), op=''):
        self.data = data
        self.grad = 0
        self._backward_fn = null_fn
        self._prev = set(parents)
        self._op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other), op='+')
        
        def _backward_add_fn():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward_fn = _backward_add_fn
        return out
    
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data * other.data, (self, other), op='*')

        def _backward_mul_fn():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        
        out._backward_fn = _backward_mul_fn
        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self): # -self
        return self * -1  
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"  

    def backward(self):
        self.grad = 1
        bts = self._gen_back_topo_seq()
        print(bts)
        for b in bts:
            b._backward_fn()

    def _gen_back_topo_seq(self):
        nodes = []
        visited = set([self])
        stack = collections.deque([self])
        while stack:
            top = stack[-1]
            deeper = False
            for p in top._prev:
                if p not in visited:
                    visited.add(p)
                    stack.append(p)
                    deeper = True
                    break
            
            if deeper: continue
            nodes.append(stack.pop())
        nodes.reverse()
        return nodes

    def relu(self):
        _relu = lambda d: 0 if d < 0 else d 
        out = Value(_relu(self.data), (self,), 'ReLU')
        
        def _backward_relu_fn():
            self.grad += (self.data > 0) * out.grad

        out._backward_fn = _backward_relu_fn
        return out