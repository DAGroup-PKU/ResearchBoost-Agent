import torch
import torch_musa  # noqa: F401
import tilelang
import tilelang.language as T


@tilelang.jit(target="musa", out_idx=[2])
def make_vector_add(n, block=256):
    @T.prim_func
    def add(
        a: T.Tensor((n,), "float32"),
        b: T.Tensor((n,), "float32"),
        out: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for i in T.Parallel(block):
                index = bx * block + i
                if index < n:
                    out[index] = a[index] + b[index]
    return add


if __name__ == "__main__":
    n = 1003
    host_index = torch.arange(n, dtype=torch.int64)
    host_a = (host_index - 500).float()
    host_b = ((host_index % 13) - 7).float() * 0.25
    kernel = make_vector_add(n)
    result = kernel(host_a.to("musa"), host_b.to("musa"))
    torch.musa.synchronize()
    torch.testing.assert_close(result.cpu(), host_a + host_b, rtol=0, atol=0)
    print(f"TILELANG_MUSA_PASS n={n} exact=1")
