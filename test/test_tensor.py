import unittest
from h2autograd import Tensor

class TestTensor(unittest.TestCase):

    def test_get_value_by_index(self):
        a = Tensor([1, 2, 3])
        self.assertEqual(a[0].data, 1)

    def test_add_tensor_broadcast(self):
        pass

    def test_add_tensor(self):
        pass

    def test_mul_one_dim_tensor(self):
        a = Tensor([1, 2, 3])
        b = Tensor([[1, 2, 3], [3, 4 ,5]])
        r = b * a
        self.assertEqual([r[0].data, r[1].data], [14, 26])
    
    def test_mul_one_dim_list(self):
        a = [1, 2, 3]
        b = Tensor([[1, 2, 3], [3, 4 ,5]])
        r = b * a
        self.assertEqual([r[0].data, r[1].data], [14, 26])

    def test_backward(self):
        pass