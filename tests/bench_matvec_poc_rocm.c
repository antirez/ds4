/* Standalone bandwidth POC for the ROCm decode matvec + MXFP4 MoE decode
 * gate/up; see rocm/ds4_rocm_poc.cuh for variants and questions. */

#include "ds4_gpu.h"

extern int ds4_gpu_rocm_matvec_poc(void);
extern int ds4_gpu_rocm_moe_poc(void);
extern int ds4_gpu_rocm_moe_gap_poc(void);

int main(int argc, char **argv) {
    if (argc > 1 && argv[1][0] == 'm') return ds4_gpu_rocm_moe_poc();
    if (argc > 1 && argv[1][0] == 'g') return ds4_gpu_rocm_moe_gap_poc();
    return ds4_gpu_rocm_matvec_poc();
}
