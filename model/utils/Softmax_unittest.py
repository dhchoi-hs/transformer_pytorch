import torch
import unittest
from Softmax import Softmax, LogSoftmax


class TestLayerNorm(unittest.TestCase):
    test_dim = -1
    test_shape = [10,512]

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_softmax(self):
        softmax = Softmax(self.test_dim)
        inp = torch.randn(self.test_shape)

        my_softmax_result = softmax(inp)
        
        torch_softmax_result = torch.nn.functional.softmax(inp, dim=self.test_dim)

        self.assertTrue(torch.isclose(my_softmax_result, torch_softmax_result).all())

    def test_log_softmax(self):
        log_softmax = LogSoftmax(self.test_dim)
        inp = torch.randn(self.test_shape)

        my_log_softmax_result = log_softmax(inp)
        
        torch_log_softmax_result = torch.nn.functional.log_softmax(inp, dim=self.test_dim)
        
        self.assertTrue(torch.isclose(my_log_softmax_result, torch_log_softmax_result).all())


if __name__ == '__main__':
    unittest.main()
