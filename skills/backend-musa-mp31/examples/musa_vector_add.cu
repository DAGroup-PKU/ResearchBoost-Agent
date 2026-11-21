#include <musa_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

void check(musaError_t status, const char* operation) {
    if (status != musaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, musaGetErrorString(status));
        std::exit(1);
    }
}

__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out,
                           int size) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < size) out[i] = a[i] + b[i];
}

int main() {
    int device_count = 0;
    check(musaGetDeviceCount(&device_count), "device count");
    if (device_count < 1) {
        std::fprintf(stderr, "no MUSA device visible\n");
        return 2;
    }

    musaDeviceProp prop{};
    check(musaGetDeviceProperties(&prop, 0), "device properties");
    std::printf(
        "ARCH name=%s capability=%d.%d mp=%d warp=%d shared_per_block=%zu "
        "shared_per_mp=%zu regs_per_block=%d l2_bytes=%d max_threads_block=%d\n",
        prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.warpSize,
        static_cast<size_t>(prop.sharedMemPerBlock),
        static_cast<size_t>(prop.sharedMemPerMultiprocessor),
        prop.regsPerBlock, prop.l2CacheSize, prop.maxThreadsPerBlock);

    constexpr int size = 1003;
    constexpr int block = 256;
    constexpr size_t bytes = size * sizeof(float);
    std::vector<float> a(size), b(size), result(size);
    for (int i = 0; i < size; ++i) {
        a[i] = float(i - 500);
        b[i] = float((i % 13) - 7) * 0.25f;
    }

    float *da = nullptr, *db = nullptr, *dout = nullptr;
    check(musaMalloc(&da, bytes), "allocate A");
    check(musaMalloc(&db, bytes), "allocate B");
    check(musaMalloc(&dout, bytes), "allocate output");
    check(musaMemcpy(da, a.data(), bytes, musaMemcpyHostToDevice), "copy A");
    check(musaMemcpy(db, b.data(), bytes, musaMemcpyHostToDevice), "copy B");

    vector_add<<<(size + block - 1) / block, block>>>(da, db, dout, size);
    check(musaGetLastError(), "launch");
    check(musaDeviceSynchronize(), "synchronize");
    check(musaMemcpy(result.data(), dout, bytes, musaMemcpyDeviceToHost), "copy output");

    for (int i = 0; i < size; ++i) {
        if (result[i] != a[i] + b[i]) {
            std::fprintf(stderr, "wrong result at %d\n", i);
            return 3;
        }
    }

    check(musaFree(dout), "free output");
    check(musaFree(db), "free B");
    check(musaFree(da), "free A");
    std::printf("MUSA_C_PASS n=%d threads=%d devices=%d exact=1\n", size, block, device_count);
    return 0;
}
