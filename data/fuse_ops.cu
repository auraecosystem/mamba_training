#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

__global__ void fused_bias_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t size,
    int64_t hidden_dim
) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= size) {
        return;
    }

    const int64_t bias_idx = idx % hidden_dim;

    const float x = input[idx] + bias[bias_idx];

    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kGeluCoeff = 0.044715f;

    const float x3 = x * x * x;
    const float inner =
        kSqrt2OverPi * (x + kGeluCoeff * x3);

    output[idx] =
        0.5f * x * (1.0f + tanhf(inner));
}

torch::Tensor fused_bias_gelu_cuda(
    torch::Tensor input,
    torch::Tensor bias
) {
    TORCH_CHECK(
        input.is_cuda(),
        "input must be a CUDA tensor"
    );

    TORCH_CHECK(
        bias.is_cuda(),
        "bias must be a CUDA tensor"
    );

    TORCH_CHECK(
        input.is_contiguous(),
        "input must be contiguous"
    );

    TORCH_CHECK(
        bias.is_contiguous(),
        "bias must be contiguous"
    );

    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32,
        "input must be float32"
    );

    TORCH_CHECK(
        bias.scalar_type() == torch::kFloat32,
        "bias must be float32"
    );

    TORCH_CHECK(
        input.dim() >= 1,
        "input must have at least one dimension"
    );

    const int64_t hidden_dim = bias.size(0);

    TORCH_CHECK(
        hidden_dim > 0,
        "bias must not be empty"
    );

    TORCH_CHECK(
        input.size(-1) == hidden_dim,
        "input last dimension must equal bias dimension"
    );

    const int64_t size = input.numel();

    auto output = torch::empty_like(input);

    constexpr int threads = 256;

    const int64_t blocks =
        (size + threads - 1) / threads;

    fused_bias_gelu_kernel<<<
        static_cast<unsigned int>(blocks),
        threads
    >>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        size,
        hidden_dim
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
