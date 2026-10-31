// 8x8 block DCT-II and its inverse on CPU with AVX2 + FMA (+ F16C).
//
// Computes in float32. Inputs may be float32, float16, bfloat16, uint8, int8, int16 or int32:
// floating inputs keep their dtype, integer inputs produce float32.
//
// SIMD runs across blocks: one __m256 holds the same pixel (or coefficient) of 8 horizontally
// adjacent blocks. Output coefficient z of those 8 blocks is then contiguous in the
// (B, C * sel, H / 8, W / 8) layout, so every store is a full vector store.
//
// Each 1D transform uses the even/odd (partial butterfly) decomposition: 36 ops instead of 64.
// The 1D matrix comes from Python, so any normalization works as long as the matrix has the
// DCT-II symmetry M[u][7 - r] = (-1)^u M[u][r].
//
//   forward(x, mat, perm, sels):  y = mat x along rows and columns of each block; channel c keeps
//                                 its first sels[c] coefficients in `perm` (zigzag) order, laid
//                                 out channel by channel: (B, C, H, W) -> (B, sum(sels), H/8, W/8).
//   inverse(y, mat, perm, sels):  x = mat^T y, i.e. the adjoint of forward with the same mat;
//                                 dropped coefficients are zero.
//
// `sels` holds one count per channel (all equal for a uniform selection), so per-channel
// selections are read and written in place, with no gather/scatter pass outside the kernel.
//
// So forward and inverse with the same matrix are each other's adjoint, which is what the
// autograd wrapper in dct/_cpu.py relies on.

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <immintrin.h>

#include <cstdint>
#include <cstring>
#include <vector>

#if !defined(__AVX2__)
#error "dct8_cpu.cpp must be compiled with AVX2, FMA and F16C enabled (/arch:AVX2 or -mavx2 -mfma -mf16c)"
#endif

namespace {

constexpr int K = 8;

inline __m256 add(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
inline __m256 sub(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
inline __m256 mul(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
inline __m256 fmadd(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }

inline void transpose8(__m256 r[8]) {
    __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]), t1 = _mm256_unpackhi_ps(r[0], r[1]);
    __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]), t3 = _mm256_unpackhi_ps(r[2], r[3]);
    __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]), t5 = _mm256_unpackhi_ps(r[4], r[5]);
    __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]), t7 = _mm256_unpackhi_ps(r[6], r[7]);
    __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    r[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    r[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    r[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    r[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    r[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    r[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    r[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    r[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// Broadcast butterfly coefficients of the 1D matrix m[u][r] (row u = frequency).
struct Coefs {
    // forward
    __m256 ee[2][2], eo[2][2], o[4][4];
    // inverse (transposed use of the same matrix)
    __m256 iee[2][2], ieo[2][2], io[4][4];

    explicit Coefs(const float* mat) {
        auto b = [](float v) { return _mm256_set1_ps(v); };
        for (int r = 0; r < 2; ++r) {
            ee[0][r] = b(mat[0 * K + r]); ee[1][r] = b(mat[4 * K + r]);
            eo[0][r] = b(mat[2 * K + r]); eo[1][r] = b(mat[6 * K + r]);
        }
        for (int k = 0; k < 4; ++k)
            for (int r = 0; r < 4; ++r) o[k][r] = b(mat[(2 * k + 1) * K + r]);
        for (int i = 0; i < 2; ++i) {
            iee[i][0] = b(mat[0 * K + i]); iee[i][1] = b(mat[4 * K + i]);
            ieo[i][0] = b(mat[2 * K + i]); ieo[i][1] = b(mat[6 * K + i]);
        }
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 4; ++k) io[i][k] = b(mat[(2 * k + 1) * K + i]);
    }
};

// ---- 1D transforms, in place on p[0], p[st], ..., p[7 * st] ----

inline void fwd_1d(__m256* p, int st, const Coefs& c) {
    __m256 x0 = p[0], x1 = p[st], x2 = p[2 * st], x3 = p[3 * st];
    __m256 x4 = p[4 * st], x5 = p[5 * st], x6 = p[6 * st], x7 = p[7 * st];
    __m256 s0 = add(x0, x7), s1 = add(x1, x6), s2 = add(x2, x5), s3 = add(x3, x4);
    __m256 d[4] = {sub(x0, x7), sub(x1, x6), sub(x2, x5), sub(x3, x4)};
    __m256 a0 = add(s0, s3), a1 = add(s1, s2), b0 = sub(s0, s3), b1 = sub(s1, s2);
    p[0]      = fmadd(c.ee[0][0], a0, mul(c.ee[0][1], a1));
    p[4 * st] = fmadd(c.ee[1][0], a0, mul(c.ee[1][1], a1));
    p[2 * st] = fmadd(c.eo[0][0], b0, mul(c.eo[0][1], b1));
    p[6 * st] = fmadd(c.eo[1][0], b0, mul(c.eo[1][1], b1));
    for (int k = 0; k < 4; ++k)
        p[(2 * k + 1) * st] = fmadd(c.o[k][3], d[3], fmadd(c.o[k][2], d[2],
                              fmadd(c.o[k][1], d[1], mul(c.o[k][0], d[0]))));
}

inline void inv_1d(__m256* p, int st, const Coefs& c) {
    __m256 y[K];
    for (int u = 0; u < K; ++u) y[u] = p[u * st];
    __m256 ee0 = fmadd(c.iee[0][1], y[4], mul(c.iee[0][0], y[0]));
    __m256 ee1 = fmadd(c.iee[1][1], y[4], mul(c.iee[1][0], y[0]));
    __m256 eo0 = fmadd(c.ieo[0][1], y[6], mul(c.ieo[0][0], y[2]));
    __m256 eo1 = fmadd(c.ieo[1][1], y[6], mul(c.ieo[1][0], y[2]));
    __m256 e[4] = {add(ee0, eo0), add(ee1, eo1), sub(ee1, eo1), sub(ee0, eo0)};
    for (int i = 0; i < 4; ++i) {
        __m256 o = fmadd(c.io[i][3], y[7], fmadd(c.io[i][2], y[5],
                   fmadd(c.io[i][1], y[3], mul(c.io[i][0], y[1]))));
        p[i * st] = add(e[i], o);
        p[(7 - i) * st] = sub(e[i], o);
    }
}

// ---- Typed load / store of 8 consecutive elements as 8 float lanes ----
//
// The transform always runs in float32; only the memory side depends on the dtype. Integer
// inputs are read directly (4x less input traffic for uint8 than a separate .float() pass),
// and float16 / bfloat16 are widened on load and rounded once on store.

struct F32 {
    using T = float;
    static __m256 load(const T* p) { return _mm256_loadu_ps(p); }
    static void store(T* p, __m256 v) { _mm256_storeu_ps(p, v); }
};

struct F16 {
    using T = uint16_t;              // IEEE half bits (c10::Half)
    static __m256 load(const T* p) { return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))); }
    static void store(T* p, __m256 v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
};

struct BF16 {
    using T = uint16_t;              // upper half of a float32 (c10::BFloat16)
    static __m256 load(const T* p) {
        __m256i w = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
        return _mm256_castsi256_ps(_mm256_slli_epi32(w, 16));
    }
    static void store(T* p, __m256 v) {
        // Round to nearest even, as c10::BFloat16 does; NaN stays a (quiet) NaN.
        __m256i b = _mm256_castps_si256(v);
        __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(b, 16), _mm256_set1_epi32(1));
        __m256i r = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(b, _mm256_set1_epi32(0x7FFF)), lsb), 16);
        __m256i nan = _mm256_castps_si256(_mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        r = _mm256_blendv_epi8(r, _mm256_set1_epi32(0x7FC0), nan);
        __m128i packed = _mm_packus_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), packed);
    }
};

struct U8 {
    using T = uint8_t;
    static __m256 load(const T* p) {
        return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
    }
};

struct I8 {
    using T = int8_t;
    static __m256 load(const T* p) {
        return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
    }
};

struct I16 {
    using T = int16_t;
    static __m256 load(const T* p) {
        return _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))));
    }
};

struct I32 {
    using T = int32_t;
    static __m256 load(const T* p) {
        return _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
    }
};

// ---- 2D on a group of 8 horizontally adjacent blocks ----

// in: top-left of the group, row stride `ld` elements (8 rows x 64 elements are read).
// Y[u * 8 + v]: coefficient (u, v) of the 8 blocks, one per lane.
template <class In>
inline void fwd_group(const typename In::T* in, int64_t ld, __m256 Y[64], const Coefs& c) {
    for (int r = 0; r < K; ++r) {
        __m256 L[8];
        for (int j = 0; j < 8; ++j) L[j] = In::load(in + r * ld + j * K);
        transpose8(L);                       // L[col] = pixel (r, col) of blocks 0..7
        for (int col = 0; col < K; ++col) Y[r * K + col] = L[col];
        fwd_1d(Y + r * K, 1, c);             // rows
    }
    for (int v = 0; v < K; ++v) fwd_1d(Y + v, K, c);   // columns
}

template <class Out>
inline void inv_group(__m256 Y[64], typename Out::T* out, int64_t ld, const Coefs& c) {
    for (int v = 0; v < K; ++v) inv_1d(Y + v, K, c);   // columns
    for (int i = 0; i < K; ++i) {
        inv_1d(Y + i * K, 1, c);             // rows
        __m256 L[8];
        for (int j = 0; j < 8; ++j) L[j] = Y[i * K + j];
        transpose8(L);                       // L[b] = row i of block b
        for (int b = 0; b < 8; ++b) Out::store(out + i * ld + b * K, L[b]);
    }
}

// Offset of each channel's first coefficient plane, and the total count N = sum(sels).
std::vector<int64_t> channel_offsets(const int64_t* sels, int64_t C, int64_t& N) {
    std::vector<int64_t> off(C);
    N = 0;
    for (int64_t c = 0; c < C; ++c) { off[c] = N; N += sels[c]; }
    return off;
}

template <class In, class Out>
void forward_impl(const torch::Tensor& x, torch::Tensor& out, const Coefs& c, const int64_t* zz,
                  const int64_t* ns, const std::vector<int64_t>& off, int64_t N) {
    using TI = typename In::T;
    using TO = typename Out::T;
    const int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const int64_t hb = H / K, wb = W / K, P = B * C;
    const TI* src = static_cast<const TI*>(x.data_ptr());
    TO* dst = static_cast<TO*>(out.data_ptr());
    const int64_t plane_out = hb * wb;

    // One task = one row of blocks (8 image rows) of one plane.
    at::parallel_for(0, P * hb, 1, [&](int64_t begin, int64_t end) {
        alignas(32) __m256 Y[64];
        alignas(32) TI tin[K * 64];
        alignas(32) TO tout[8];
        for (int64_t t = begin; t < end; ++t) {
            const int64_t pl = t / hb, p = t % hb;
            const int64_t sel = ns[pl % C];
            const TI* row = src + (pl * H + p * K) * W;
            TO* obase = dst + ((pl / C) * N + off[pl % C]) * plane_out + p * wb;
            int64_t q = 0;
            for (; q + 8 <= wb; q += 8) {
                fwd_group<In>(row + q * K, W, Y, c);
                for (int64_t z = 0; z < sel; ++z) Out::store(obase + z * plane_out + q, Y[zz[z]]);
            }
            if (q < wb) {                    // tail: < 8 blocks, go through a zero-padded buffer
                const int64_t n = wb - q;
                std::memset(tin, 0, sizeof(tin));
                for (int r = 0; r < K; ++r)
                    std::memcpy(tin + r * 64, row + r * W + q * K, n * K * sizeof(TI));
                fwd_group<In>(tin, 64, Y, c);
                for (int64_t z = 0; z < sel; ++z) {
                    Out::store(tout, Y[zz[z]]);
                    std::memcpy(obase + z * plane_out + q, tout, n * sizeof(TO));
                }
            }
        }
    });
}

template <class In, class Out>
void inverse_impl(const torch::Tensor& y, torch::Tensor& out, const Coefs& c, const int64_t* zz,
                  const int64_t* ns, const std::vector<int64_t>& off) {
    using TI = typename In::T;
    using TO = typename Out::T;
    const int64_t B = y.size(0), N = y.size(1), hb = y.size(2), wb = y.size(3);
    const int64_t C = out.size(1), H = hb * K, W = wb * K, P = B * C;
    const TI* src = static_cast<const TI*>(y.data_ptr());
    TO* dst = static_cast<TO*>(out.data_ptr());
    const int64_t plane_in = hb * wb;

    at::parallel_for(0, P * hb, 1, [&](int64_t begin, int64_t end) {
        alignas(32) __m256 Y[64];
        alignas(32) TO tout[K * 64];
        alignas(32) TI tin[8];
        for (int64_t t = begin; t < end; ++t) {
            const int64_t pl = t / hb, p = t % hb;
            const int64_t sel = ns[pl % C];
            const TI* ibase = src + ((pl / C) * N + off[pl % C]) * plane_in + p * wb;
            TO* row = dst + (pl * H + p * K) * W;
            int64_t q = 0;
            for (; q + 8 <= wb; q += 8) {
                for (int z = 0; z < 64; ++z) Y[z] = _mm256_setzero_ps();
                for (int64_t z = 0; z < sel; ++z) Y[zz[z]] = In::load(ibase + z * plane_in + q);
                inv_group<Out>(Y, row + q * K, W, c);
            }
            if (q < wb) {
                const int64_t n = wb - q;
                for (int z = 0; z < 64; ++z) Y[z] = _mm256_setzero_ps();
                for (int64_t z = 0; z < sel; ++z) {
                    std::memset(tin, 0, sizeof(tin));
                    std::memcpy(tin, ibase + z * plane_in + q, n * sizeof(TI));
                    Y[zz[z]] = In::load(tin);
                }
                inv_group<Out>(Y, tout, 64, c);
                for (int r = 0; r < K; ++r)
                    std::memcpy(row + r * W + q * K, tout + r * 64, n * K * sizeof(TO));
            }
        }
    });
}

// Output dtype: floating inputs keep theirs, integer inputs produce float32 (as in dct.functional).
torch::ScalarType out_dtype(torch::ScalarType t) {
    return (t == torch::kHalf || t == torch::kBFloat16) ? t : torch::kFloat32;
}

// Calls f(In{}, Out{}) for the input dtype.
template <class F>
void dispatch(torch::ScalarType t, F&& f) {
    switch (t) {
        case torch::kFloat32:  return f(F32{}, F32{});
        case torch::kHalf:     return f(F16{}, F16{});
        case torch::kBFloat16: return f(BF16{}, BF16{});
        case torch::kByte:     return f(U8{}, F32{});
        case torch::kChar:     return f(I8{}, F32{});
        case torch::kShort:    return f(I16{}, F32{});
        case torch::kInt:      return f(I32{}, F32{});
        default: TORCH_CHECK(false, "unsupported dtype ", t);
    }
}

bool supported_dtype(torch::ScalarType s) {
    return s == torch::kFloat32 || s == torch::kHalf || s == torch::kBFloat16 || s == torch::kByte ||
           s == torch::kChar || s == torch::kShort || s == torch::kInt;
}

void check(const torch::Tensor& t, const torch::Tensor& mat, const torch::Tensor& perm, const torch::Tensor& sels) {
    TORCH_CHECK(t.device().is_cpu() && t.dim() == 4 && supported_dtype(t.scalar_type()),
                "expected a 4D CPU tensor of dtype float32, float16, bfloat16, uint8, int8, int16 or int32");
    TORCH_CHECK(mat.device().is_cpu() && mat.scalar_type() == torch::kFloat32 && mat.numel() == 64,
                "mat must be an 8x8 float32 CPU tensor");
    TORCH_CHECK(perm.device().is_cpu() && perm.scalar_type() == torch::kInt64 && perm.numel() == 64
                && perm.is_contiguous(), "perm must be 64 contiguous int64 values on the CPU");
    TORCH_CHECK(sels.device().is_cpu() && sels.scalar_type() == torch::kInt64 && sels.dim() == 1
                && sels.is_contiguous(), "sels must be a contiguous 1D int64 CPU tensor");
    const int64_t* ns = sels.data_ptr<int64_t>();
    for (int64_t c = 0; c < sels.numel(); ++c)
        TORCH_CHECK(ns[c] >= 1 && ns[c] <= 64, "every selection count must be in [1, 64]");
}

}  // namespace

torch::Tensor dct8_forward(torch::Tensor x, torch::Tensor mat, torch::Tensor perm, torch::Tensor sels) {
    check(x, mat, perm, sels);
    TORCH_CHECK(x.size(2) % K == 0 && x.size(3) % K == 0, "H and W must be divisible by 8");
    TORCH_CHECK(sels.numel() == x.size(1), "sels must have one entry per input channel");
    x = x.contiguous();
    int64_t N;
    const auto off = channel_offsets(sels.data_ptr<int64_t>(), x.size(1), N);
    auto out = torch::empty({x.size(0), N, x.size(2) / K, x.size(3) / K},
                            x.options().dtype(out_dtype(x.scalar_type())));
    const auto mat_c = mat.contiguous();
    const Coefs c(mat_c.data_ptr<float>());
    dispatch(x.scalar_type(), [&](auto in, auto o) {
        forward_impl<decltype(in), decltype(o)>(x, out, c, perm.data_ptr<int64_t>(),
                                                sels.data_ptr<int64_t>(), off, N);
    });
    return out;
}

torch::Tensor dct8_inverse(torch::Tensor y, torch::Tensor mat, torch::Tensor perm, torch::Tensor sels) {
    check(y, mat, perm, sels);
    TORCH_CHECK(sels.numel() >= 1, "sels must not be empty");
    int64_t total;
    const auto off = channel_offsets(sels.data_ptr<int64_t>(), sels.numel(), total);
    TORCH_CHECK(y.size(1) == total, "channels must equal sum(sels)");
    y = y.contiguous();
    auto out = torch::empty({y.size(0), sels.numel(), y.size(2) * K, y.size(3) * K},
                            y.options().dtype(out_dtype(y.scalar_type())));
    const auto mat_c = mat.contiguous();
    const Coefs c(mat_c.data_ptr<float>());
    dispatch(y.scalar_type(), [&](auto in, auto o) {
        inverse_impl<decltype(in), decltype(o)>(y, out, c, perm.data_ptr<int64_t>(),
                                                sels.data_ptr<int64_t>(), off);
    });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dct8_forward, py::call_guard<py::gil_scoped_release>());
    m.def("inverse", &dct8_inverse, py::call_guard<py::gil_scoped_release>());
}
