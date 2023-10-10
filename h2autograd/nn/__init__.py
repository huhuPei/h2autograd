from h2autograd import Value

class Module:
    def __init__(self):
        pass

class Relu(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Value):
        return x.relu()

class Linear(Module):
    def __init__(self):
        super().__init__()
