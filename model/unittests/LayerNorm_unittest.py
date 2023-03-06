import unittest
import torch
from LayerNorm import LayerNorm
import numpy as np
import math


class TestLayerNorm(unittest.TestCase):
    test_shape = [10,512]

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_equal(self):
        layerNorm = LayerNorm(512)
        inp = torch.randn(self.test_shape)
        outp = layerNorm(inp)
        
        mean = outp.mean(-1).detach().numpy()
        std = outp.std(-1).detach().numpy()

        r_mean = np.array([0.]*10)
        r_std = np.array([1.]*10)

        self.assertTrue(np.isclose(mean, r_mean, atol=1e-6).all())
        self.assertTrue(np.isclose(std, r_std, atol=1e-6).all())


if __name__ == '__main__':
    unittest.main()
