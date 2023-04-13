import unittest
import torch
from torch import nn
from model.utils.Linear import Linear


print(__name__)
print(__file__)
class TestLinear(unittest.TestCase):
    in_feat = 10
    out_feat = 5

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_linear(self):
        for i in range(10):
            with self.subTest(i=i):
                custom_w = torch.randn([self.out_feat, self.in_feat])
                custom_b = torch.randn(self.out_feat)

                l = Linear(self.in_feat, self.out_feat)
                l2 = nn.Linear(self.in_feat, self.out_feat)
                p = l.state_dict()
                p2 = l2.state_dict()
                p['weights'] = custom_w.T
                p['bias'] = custom_b
                p2['weight'] = custom_w
                p2['bias'] = custom_b
                l.load_state_dict(p)
                l2.load_state_dict(p2)
                
                test_input = torch.randn([20, self.in_feat])
                output = l(test_input)
                output2 = l2(test_input)

                self.assertTrue(torch.isclose(output, output2, 0.0001).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
