// Inference-only FP32 reference recurrence. Each head owns its state copy.
// No fast-math or fused multiply-add: retain the reference update boundaries.
#include <vector>
#include <cstdint>

extern "C" void rwkv_wkv_cpu(
    const float* r, const float* decay, const float* k, const float* v,
    const float* a, const float* b, float* state, float* output,
    int batch, int steps, int heads, int size, int threads) {
    const int channels = heads * size;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int bh = 0; bh < batch * heads; ++bh) {
        const int bi = bh / heads, hi = bh % heads;
        float* s = state + int64_t(bh) * size * size;
        std::vector<float> sa(size);
        for (int t = 0; t < steps; ++t) {
            const int64_t offset = (int64_t(bi) * steps + t) * channels + hi * size;
            for (int i = 0; i < size; ++i) {
                float dot = 0;
                #pragma omp simd reduction(+:dot)
                for (int j = 0; j < size; ++j) dot += s[i * size + j] * a[offset + j];
                sa[i] = dot;
            }
            for (int i = 0; i < size; ++i) {
                #pragma omp simd
                for (int j = 0; j < size; ++j) {
                    const float decayed = s[i * size + j] * decay[offset + j];
                    const float correction = sa[i] * b[offset + j];
                    const float outer = v[offset + i] * k[offset + j];
                    s[i * size + j] = (decayed + correction) + outer;
                }
                float dot = 0;
                #pragma omp simd reduction(+:dot)
                for (int j = 0; j < size; ++j) dot += s[i * size + j] * r[offset + j];
                output[offset + i] = dot;
            }
        }
    }
}
