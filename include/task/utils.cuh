

__global__ void flush_l2_cache(float* buffer, size_t size, float* sink) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float acc = 0.0f;
    for (size_t i = idx; i < size; i += stride) {
        acc += __ldcg(&buffer[i]);
    }
    if (idx == 0) {
        *sink += acc;
    }
}