"""The CPU oracle must retain the released kernels' block boundaries."""
import unittest

import torch

import deepseek41_native_reference as reference


class SourceAccumulationTests(unittest.TestCase):
    def test_scaled_blocks_round_before_next_block(self):
        # Source order is (1 + 2**25) - 2**25 == 0 in FP32. A single
        # BLAS dot may cancel the large terms first and incorrectly return 1.
        x = torch.ones(1, 96).to(torch.float8_e4m3fn)
        scales = torch.ones(1, 3)
        ws = torch.tensor([[1., 2**25, 2**25]])
        weights = torch.zeros(32, 96)
        weights[:, 0], weights[:, 33], weights[:, 65] = 1, 1, -1
        packed = torch.zeros(1, 48, dtype=torch.uint8)
        packed[0, 0], packed[0, 16], packed[0, 32] = 2, 2 << 4, 10 << 4
        for kind in ('fp8', 'fp4'):
            with self.subTest(kind=kind):
                result = reference.fp8_gemm(x, scales, weights.to(x.dtype), ws) if kind == 'fp8' else reference.fp4_gemm(
                    x, scales, packed.view(torch.float4_e2m1fn_x2), ws)
                self.assertTrue(torch.equal(result, torch.zeros_like(result)), result)


if __name__ == '__main__':
    unittest.main()
