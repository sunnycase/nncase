#include "../../include/nncase/ntt/ukernels.h"
#include "nncase/ntt/apply.h"
#include "nncase/ntt/arch/riscv64/arch_types.h"
#include "nncase/ntt/kernels/transpose.h"
#include "nncase/ntt/kernels/unpack.h"
#include "nncase/ntt/primitive_ops.h"
#include "nncase/ntt/shape.h"
#include "nncase/ntt/tensor.h"
#include "nncase/ntt/vector.h"
#include "ntt_test.h"
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <nncase/ntt/ntt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

using namespace nncase;

template <size_t M, size_t K, size_t N>
void gemm_naive(ntt::fixed_tensor<float, M, K> &a,
                ntt::fixed_tensor<float, K, N> &b,
                ntt::fixed_tensor<float, M, N> &c) {
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            c(m, n) = 0;
            for (size_t k = 0; k < K; k++) {
                c(m, n) += a(m, k) * b(k, n);
            }
        }
    }
}

template <size_t M, size_t K, size_t N> void benchmark_ntt_matmul_pack_NONE() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = 3000;

    ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
    ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
    ntt::tensor<float, ntt::fixed_shape<M, N>> tc;
    std::iota(ta.elements().begin(), ta.elements().end(), 0.f);
    std::iota(tb.elements().begin(), tb.elements().end(), 0.f);

    for (size_t i = 0; i < warmup_num; i++)
        gemm_naive(ta, tb, tc);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < run_num; i++) {
        gemm_naive(ta, tb, tc);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    asm volatile("" ::"g"(tc));

    auto ops = M * N * K * 2;
    // auto t = get_time(&start, &end) / run_num;
    auto t =
        std::chrono::duration<double, std::ratio<1>>(stop - start).count() /
        run_num;
    std::cout << std::setprecision(0) << std::fixed << M << ", "
              << std::setprecision(1) << std::fixed << ops / t * 1e-9
              << std::endl;
}

#define GEMM_VERSION 7
#define CONCAT_(x, y) x##y
#define GEMM_CONCAT_(y) CONCAT_(gemm_level, y)
#define GEMM_NAME GEMM_CONCAT_(GEMM_VERSION)

#if __x86_64__
#include <immintrin.h>

template <size_t M, size_t K, size_t N>
void gemm_level1(ntt::fixed_tensor<ntt::vector<float, 8>, M, K> &a,
                 ntt::fixed_tensor<float, K, N> &b,
                 ntt::fixed_tensor<ntt::vector<float, 8>, M, N> &c) {
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            auto val = _mm256_set1_ps(0);
            for (size_t k = 0; k < K; k++) {
                auto b_cast = _mm256_broadcast_ss(&b(k, n));
                val = _mm256_fmadd_ps(a(m, k), b_cast, val);
            }
            c(m, n) = val;
        }
    }
}

template <size_t M, size_t K, size_t N>
void gemm_level2(ntt::fixed_tensor<ntt::vector<float, 8>, M, K> &a,
                 ntt::fixed_tensor<float, K, N> &b,
                 ntt::fixed_tensor<ntt::vector<float, 8>, M, N> &c) {
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            auto val = _mm256_set1_ps(0);
            for (size_t k1 = 0; k1 < K; k1 += 8) {
                for (size_t k = k1; k < k1 + 8; k++) {
                    auto b_cast = _mm256_broadcast_ss(&b(k, n));
                    val = _mm256_fmadd_ps(a(m, k), b_cast, val);
                }
            }
            c(m, n) = val;
        }
    }
}

template <size_t M, size_t K, size_t N>
void gemm_level3(ntt::fixed_tensor<ntt::vector<float, 8>, M, K> &a,
                 ntt::fixed_tensor<float, K, N> &b,
                 ntt::fixed_tensor<ntt::vector<float, 8>, M, N> &c) {
    for (size_t m1 = 0; m1 < M; m1 += 2) {
        for (size_t n1 = 0; n1 < N; n1 += 4) {
            __m256 c0[2][4];
            __m256 a0[2];
            __m256 b0[4];
            ntt::apply(ntt::fixed_shape<2, 4>{}, [&](auto index) {
                c0[index[0]][index[1]] = _mm256_set1_ps(0);
            });
            for (size_t k = 0; k < K; k++) {
                ntt::apply(ntt::fixed_shape<2>{}, [&](auto index) {
                    a0[index[0]] = a(m1 + index[0], k);
                });
                ntt::apply(ntt::fixed_shape<4>{}, [&](auto index) {
                    b0[index[0]] = _mm256_broadcast_ss(&b(k, n1 + index[0]));
                });
                for (size_t m0 = 0; m0 < 2; m0++) {
                    for (size_t n0 = 0; n0 < 4; n0++) {
                        c0[m0][n0] =
                            _mm256_fmadd_ps(a0[m0], b0[n0], c0[m0][n0]);
                    }
                }
            }
            ntt::apply(ntt::fixed_shape<2, 4>{}, [&](auto index) {
                c(m1 + index[0], n1 + index[1]) = c0[index[0]][index[1]];
            });
        }
    }
}

template <size_t M, size_t K, size_t N>
void gemm_level4(ntt::fixed_tensor<ntt::vector<float, 8>, M, K> &a,
                 ntt::fixed_tensor<float, N, K> &b,
                 ntt::fixed_tensor<ntt::vector<float, 8>, M, N> &c) {
    for (size_t m1 = 0; m1 < M; m1 += 2) {
        for (size_t n1 = 0; n1 < N; n1 += 4) {
            __m256 c0[2][4];
            __m256 a0[2];
            __m256 b0[4];
            ntt::apply(ntt::fixed_shape<2, 4>{}, [&](auto index) {
                c0[index[0]][index[1]] = _mm256_set1_ps(0);
            });
            for (size_t k = 0; k < K; k++) {
                ntt::apply(ntt::fixed_shape<2>{}, [&](auto index) {
                    a0[index[0]] = a(m1 + index[0], k);
                });
                ntt::apply(ntt::fixed_shape<4>{}, [&](auto index) {
                    b0[index[0]] = _mm256_broadcast_ss(&b(n1 + index[0], k));
                });
                for (size_t m0 = 0; m0 < 2; m0++) {
                    for (size_t n0 = 0; n0 < 4; n0++) {
                        c0[m0][n0] =
                            _mm256_fmadd_ps(a0[m0], b0[n0], c0[m0][n0]);
                    }
                }
            }
            ntt::apply(ntt::fixed_shape<2, 4>{}, [&](auto index) {
                c(m1 + index[0], n1 + index[1]) = c0[index[0]][index[1]];
            });
        }
    }
}

template <bool Acc, class TC>
__attribute__((always_inline)) inline void
gemm_level5_l2(const ntt::vector<float, 8> *a, const float *b, TC &&c, size_t M,
               size_t K, size_t N) {
    const ntt::vector<float, 8> *pa;
    const float *pb;
    for (size_t m1 = 0; m1 < M; m1 += 2) {
        for (size_t n1 = 0; n1 < N; n1 += 4) {
            __m256 c0[2][4];
            __m256 a0[2];
            __m256 b0[4];
            ntt::apply(ntt::fixed_shape<2, 4>{}, [&](auto index) {
                c0[index[0]][index[1]] =
                    Acc ? (__m256)c(m1 + index[0], n1 + index[1])
                        : _mm256_set1_ps(0);
            });
            pa = a + m1 * 256;
            pb = b + n1 * 256;
            for (size_t k = 0; k < K; k++) {
                a0[0] = *pa++;
                a0[1] = *pa++;
                b0[0] = _mm256_broadcast_ss(pb++);
                b0[1] = _mm256_broadcast_ss(pb++);
                b0[2] = _mm256_broadcast_ss(pb++);
                b0[3] = _mm256_broadcast_ss(pb++);
                for (size_t m0 = 0; m0 < 2; m0++) {
                    for (size_t n0 = 0; n0 < 4; n0++) {
                        c0[m0][n0] =
                            _mm256_fmadd_ps(a0[m0], b0[n0], c0[m0][n0]);
                    }
                }
            }
            ntt::apply(ntt::fixed_shape<2, 4>{}, [&](auto index) {
                c(m1 + index[0], n1 + index[1]) = c0[index[0]][index[1]];
            });
        }
    }
}

template <size_t M, size_t K, size_t N>
void gemm_level5(ntt::fixed_tensor<ntt::vector<float, 8>, M, K> &a,
                 ntt::fixed_tensor<float, N, K> &b,
                 ntt::fixed_tensor<ntt::vector<float, 8>, M, N> &c) {
    constexpr size_t m2_tile = 32;
    constexpr size_t n2_tile = 128;
    constexpr size_t k2_tile = 256;

    alignas(4096) static ntt::fixed_tensor<ntt::vector<float, 8>, m2_tile / 2,
                                           k2_tile, 2>
        a2_packed;
    alignas(4096) static ntt::fixed_tensor<
        ntt::fixed_tensor<float, n2_tile / 4, k2_tile, 4>,
        ntt::ceil_div(N, n2_tile), ntt::ceil_div(K, k2_tile)>
        b2_packs;

    for (size_t m2 = 0; m2 < M; m2 += m2_tile) {
        auto real_m2 = std::min(m2_tile, M - m2);
        for (size_t k2 = 0; k2 < K; k2 += k2_tile) {
            auto real_k2 = std::min(k2_tile, K - k2);
            auto a2 = a.view(ntt::make_ranked_shape(m2, k2),
                             ntt::make_ranked_shape(real_m2, real_k2));
            for (size_t ma = 0; ma < real_m2; ma += 2) {
                for (size_t ka = 0; ka < real_k2; ka++) {
                    a2_packed(ma / 2, ka, 0) = a2(ma, ka);
                    a2_packed(ma / 2, ka, 1) = a2(ma + 1, ka);
                }
            }
            for (size_t n2 = 0; n2 < N; n2 += n2_tile) {
                auto real_n2 = std::min(n2_tile, N - n2);
                auto b2 = b.view(ntt::make_ranked_shape(n2, k2),
                                 ntt::make_ranked_shape(real_n2, real_k2));
                auto c2 = c.view(ntt::make_ranked_shape(m2, n2),
                                 ntt::make_ranked_shape(real_m2, real_n2));
                auto &b2_packed = b2_packs(n2 / n2_tile, k2 / k2_tile);
                if (m2 == 0) {
                    for (size_t nb = 0; nb < real_n2; nb += 4) {
                        for (size_t ka = 0; ka < real_k2; ka++) {
                            b2_packed(nb / 4, ka, 0) = b2(nb, ka);
                            b2_packed(nb / 4, ka, 1) = b2(nb + 1, ka);
                            b2_packed(nb / 4, ka, 2) = b2(nb + 2, ka);
                            b2_packed(nb / 4, ka, 3) = b2(nb + 3, ka);
                        }
                    }
                }
                if (k2 != 0) {
                    gemm_level5_l2<true>(a2_packed.elements().data(),
                                         b2_packed.elements().data(), c2,
                                         real_m2, real_k2, real_n2);
                } else {
                    gemm_level5_l2<false>(a2_packed.elements().data(),
                                          b2_packed.elements().data(), c2,
                                          real_m2, real_k2, real_n2);
                }
            }
        }
    }
}

// template <bool Acc, class TA, class TB, class TC>
// void gemm_level5_l1(TA &a, TB &b, TC &c) {
//     const auto M = a.shape()[0];
//     const auto N = b.shape()[1];
//     const auto K = a.shape()[1];

//     for (size_t m1 = 0; m1 < M; m1 += 2) {
//         for (size_t n1 = 0; n1 < N; n1 += 4) {
//             __m256 c0[2][4];
//             __m256 a0[2];
//             __m256 b0[4];
//             ntt::apply(ntt::fixed_shape<2, 4>{}, [&](auto index) {
//                 c0[index[0]][index[1]] =
//                     Acc ? (__m256)c(m1 + index[0], n1 + index[1])
//                         : _mm256_set1_ps(0);
//             });
//             _mm_prefetch(&c(m1 + 0, n1 + 4 + 0), _MM_HINT_T1);
//             _mm_prefetch(&c(m1 + 0, n1 + 4 + 2), _MM_HINT_T1);
//             _mm_prefetch(&c(m1 + 1, n1 + 4 + 0), _MM_HINT_T1);
//             _mm_prefetch(&c(m1 + 1, n1 + 4 + 2), _MM_HINT_T1);
//             for (size_t k = 0; k < K; k++) {
//                 ntt::apply(ntt::fixed_shape<2>{}, [&](auto index) {
//                     a0[index[0]] = a(m1 + index[0], k);
//                 });
//                 _mm_prefetch(&a(m1 + 0, k + 1), _MM_HINT_T1);
//                 ntt::apply(ntt::fixed_shape<4>{}, [&](auto index) {
//                     b0[index[0]] = _mm256_broadcast_ss(&b(k, n1 + index[0]));
//                 });
//                 _mm_prefetch(&b(k + 1, n1), _MM_HINT_T1);
//                 for (size_t m0 = 0; m0 < 2; m0++) {
//                     for (size_t n0 = 0; n0 < 4; n0++) {
//                         c0[m0][n0] =
//                             _mm256_fmadd_ps(a0[m0], b0[n0], c0[m0][n0]);
//                     }
//                 }
//             }
//             ntt::apply(ntt::fixed_shape<2, 4>{}, [&](auto index) {
//                 c(m1 + index[0], n1 + index[1]) = c0[index[0]][index[1]];
//             });
//         }
//     }
// }

template <bool Acc, class TA, class TB, class TC>
__attribute__((always_inline)) inline void
gemm_level6_l2(TA &a, TB &b, TC &c, std::function<void()> &prefetcher) {
    const auto M = a.shape()[0];
    const auto N = b.shape()[0];
    const auto K = a.shape()[1];

    for (size_t m1 = 0; m1 < M; m1 += 2) {
        for (size_t n1 = 0; n1 < N; n1 += 4) {
            __m256 c0[2][4];
            __m256 a0[2];
            __m256 b0[4];
            ntt::apply(ntt::fixed_shape<2, 4>{}, [&](auto index) {
                c0[index[0]][index[1]] =
                    Acc ? (__m256)c(m1 + index[0], n1 + index[1])
                        : _mm256_set1_ps(0);
            });
            _mm_prefetch(&c(m1 + 0, n1 + 4 + 0), _MM_HINT_T0);
            _mm_prefetch(&c(m1 + 1, n1 + 4 + 0), _MM_HINT_T0);
            for (size_t k = 0; k < K; k++) {
                if (k == 250 && m1 + 2 >= M && n1 + 4 >= N && prefetcher) {
                    // prefetcher();
                }
                ntt::apply(ntt::fixed_shape<2>{}, [&](auto index) {
                    a0[index[0]] = a(m1 + index[0], k);
                });
                _mm_prefetch(&a(m1 + 0, k + 4), _MM_HINT_T0);
                _mm_prefetch(&b(k + 4, n1), _MM_HINT_T0);
                ntt::apply(ntt::fixed_shape<4>{}, [&](auto index) {
                    b0[index[0]] = _mm256_broadcast_ss(&b(n1 + index[0], k));
                });
                for (size_t m0 = 0; m0 < 2; m0++) {
                    for (size_t n0 = 0; n0 < 4; n0++) {
                        c0[m0][n0] =
                            _mm256_fmadd_ps(a0[m0], b0[n0], c0[m0][n0]);
                    }
                }
            }

            ntt::apply(ntt::fixed_shape<2, 4>{}, [&](auto index) {
                c(m1 + index[0], n1 + index[1]) = c0[index[0]][index[1]];
                _mm_prefetch(&c(m1 + index[0], n1 + index[1]), _MM_HINT_T2);
                //_mm256_stream_ps((float *)&c(m1 + index[0], n1 + index[1]),
                //                 c0[index[0]][index[1]]);
            });
        }
    }
}

template <size_t M, size_t K, size_t N>
void gemm_level6(ntt::fixed_tensor<ntt::vector<float, 8>, M, K> &a,
                 ntt::fixed_tensor<float, N, K> &b,
                 ntt::fixed_tensor<ntt::vector<float, 8>, M, N> &c) {
    constexpr size_t m2_tile = 8;
    constexpr size_t n2_tile = 32;
    constexpr size_t k2_tile = 64;

    static ntt::fixed_tensor<ntt::vector<float, 8>, m2_tile, k2_tile> a2_packed;
    static ntt::fixed_tensor<ntt::fixed_tensor<float, n2_tile, k2_tile>,
                             ntt::ceil_div(N, n2_tile),
                             ntt::ceil_div(K, k2_tile)>
        b2_packs;

    for (size_t m2 = 0; m2 < M; m2 += m2_tile) {
        auto real_m2 = std::min(m2_tile, M - m2);
        for (size_t k2 = 0; k2 < K; k2 += k2_tile) {
            auto real_k2 = std::min(k2_tile, K - k2);
            auto a2 = a.view(ntt::make_ranked_shape(m2, k2),
                             ntt::make_ranked_shape(real_m2, real_k2));
            ntt::apply(a2.shape(),
                       [&](auto index) { a2_packed(index) = a2(index); });
            for (size_t n2 = 0; n2 < N; n2 += n2_tile) {
                auto real_n2 = std::min(n2_tile, N - n2);
                auto b2 = b.view(ntt::make_ranked_shape(n2, k2),
                                 ntt::make_ranked_shape(real_n2, real_k2));
                auto c2 = c.view(ntt::make_ranked_shape(m2, n2),
                                 ntt::make_ranked_shape(real_m2, real_n2));
                auto &b2_packed = b2_packs(n2 / n2_tile, k2 / k2_tile);
                if (m2 == 0) {
                    ntt::apply(b2.shape(), [&](auto index) {
                        b2_packed(index) = b2(index);
                    });
                }
                if (k2 != 0) {
                    gemm_level5_l2<true>(a2_packed, b2_packed, c2, real_m2,
                                         real_k2, real_n2);
                } else {
                    gemm_level5_l2<false>(a2_packed, b2_packed, c2, real_m2,
                                          real_k2, real_n2);
                }
            }
        }
    }

    //     for (size_t m2 = 0; m2 < M; m2 += m2_tile) {
    //         auto real_m2 = std::min(m2_tile, M - m2);
    //         for (size_t k2 = 0; k2 < K; k2 += k2_tile) {
    //             auto real_k2 = std::min(k2_tile, K - k2);
    //             for (size_t n2 = 0; n2 < N; n2 += n2_tile) {
    //                 auto real_n2 = std::min(n2_tile, N - n2);
    //                 auto a2 = a.view(ntt::make_ranked_shape(m2, k2),
    //                                  ntt::make_ranked_shape(real_m2,
    //                                  real_k2));
    //                 auto b2 = b.view(ntt::make_ranked_shape(n2, k2),
    //                                  ntt::make_ranked_shape(real_n2,
    //                                  real_k2));
    //                 auto c2 = c.view(ntt::make_ranked_shape(m2, n2),
    //                                  ntt::make_ranked_shape(real_m2,
    //                                  real_n2));
    // #if 1
    //                 std::function<void()> prefetcher;
    //                 if (n2 + n2_tile >= N) {
    //                     if (k2 + k2_tile >= K) {
    // #pragma GCC diagnostic push
    // #pragma GCC diagnostic ignored "-Warray-bounds"
    //                         prefetcher = [&] {
    //                             return;
    //                             if (M > m2_tile) {
    //                                 _mm_prefetch(&c(m2 + m2_tile + 0, 0),
    //                                              _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + m2_tile + 0, 1),
    //                                              _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + m2_tile + 0, 2),
    //                                              _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + m2_tile + 0, 3),
    //                                              _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + m2_tile + 1, 0),
    //                                              _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + m2_tile + 1, 1),
    //                                              _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + m2_tile + 1, 2),
    //                                              _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + m2_tile + 1, 3),
    //                                              _MM_HINT_T0);

    //                                 _mm_prefetch(&a(m2 + m2_tile + 0, 0),
    //                                              _MM_HINT_T0);
    //                                 _mm_prefetch(&a(m2 + m2_tile + 1, 0),
    //                                              _MM_HINT_T0);

    //                                 _mm_prefetch(&b(0, 0), _MM_HINT_T0);
    //                                 _mm_prefetch(&b(1, 0), _MM_HINT_T0);
    //                                 _mm_prefetch(&b(2, 0), _MM_HINT_T0);
    //                                 _mm_prefetch(&b(3, 0), _MM_HINT_T0);
    //                             }
    //                         };
    // #pragma GCC diagnostic pop
    //                     } else {
    //                         prefetcher = [&] {
    //                             return;
    //                             if (K > k2_tile) {
    //                                 _mm_prefetch(&c(m2 + 0, 0), _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + 0, 1), _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + 0, 2), _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + 0, 3), _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + 1, 0), _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + 1, 1), _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + 1, 2), _MM_HINT_T0);
    //                                 _mm_prefetch(&c(m2 + 1, 3), _MM_HINT_T0);

    //                                 _mm_prefetch(&a(m2 + 0, k2 + k2_tile),
    //                                              _MM_HINT_T0);
    //                                 _mm_prefetch(&a(m2 + 1, k2 + k2_tile),
    //                                              _MM_HINT_T0);

    //                                 _mm_prefetch(&b(0, k2 + k2_tile),
    //                                 _MM_HINT_T0); _mm_prefetch(&b(1, k2 +
    //                                 k2_tile), _MM_HINT_T0);
    //                                 _mm_prefetch(&b(2, k2 + k2_tile),
    //                                 _MM_HINT_T0); _mm_prefetch(&b(3, k2 +
    //                                 k2_tile), _MM_HINT_T0);
    //                             }
    //                         };
    //                     }
    //                 } else {
    //                     prefetcher = [&] {
    //                         _mm_prefetch(&c(m2 + 0, n2 + n2_tile + 0),
    //                         _MM_HINT_T0); _mm_prefetch(&c(m2 + 0, n2 +
    //                         n2_tile + 1), _MM_HINT_T0); _mm_prefetch(&c(m2 +
    //                         0, n2 + n2_tile + 2), _MM_HINT_T0);
    //                         _mm_prefetch(&c(m2 + 0, n2 + n2_tile + 3),
    //                         _MM_HINT_T0); _mm_prefetch(&c(m2 + 1, n2 +
    //                         n2_tile + 0), _MM_HINT_T0); _mm_prefetch(&c(m2 +
    //                         1, n2 + n2_tile + 1), _MM_HINT_T0);
    //                         _mm_prefetch(&c(m2 + 1, n2 + n2_tile + 2),
    //                         _MM_HINT_T0); _mm_prefetch(&c(m2 + 1, n2 +
    //                         n2_tile + 3), _MM_HINT_T0);

    //                         _mm_prefetch(&a(m2 + 0, k2), _MM_HINT_T0);
    //                         _mm_prefetch(&a(m2 + 1, k2), _MM_HINT_T0);

    //                         _mm_prefetch(&b(n2 + n2_tile + 0, k2),
    //                         _MM_HINT_T0); _mm_prefetch(&b(n2 + n2_tile + 1,
    //                         k2), _MM_HINT_T0); _mm_prefetch(&b(n2 + n2_tile +
    //                         2, k2), _MM_HINT_T0); _mm_prefetch(&b(n2 +
    //                         n2_tile + 3, k2), _MM_HINT_T0);
    //                     };
    //                 }
    // #endif
    //                 if (k2 != 0) {
    //                     gemm_level6_l2<true>(a2, b2, c2, prefetcher);
    //                 } else {
    //                     gemm_level6_l2<false>(a2, b2, c2, prefetcher);
    //                 }
    //             }
    //         }
    //     }
}
#else
#include <riscv_vector.h>

template <size_t M, size_t K, size_t N>
void gemm_level4(ntt::fixed_tensor<ntt::vector<float, 8>, M, K> &a,
                 ntt::fixed_tensor<float, N, K> &b,
                 ntt::fixed_tensor<ntt::vector<float, 8>, M, N> &c) {
    constexpr size_t vl = NTT_VLEN / (sizeof(float) * 8);
    for (size_t m1 = 0; m1 < M; m1 += 2) {
        for (size_t n1 = 0; n1 < N; n1 += 4) {
            fixed_vfloat32m1_t c0[2][4];
            fixed_vfloat32m1_t a0[2];
            float b0[4];
            ntt::apply(ntt::fixed_shape<2, 4>{}, [&](auto index) {
                c0[index[0]][index[1]] = fixed_vfloat32m1_t{};
            });
            for (size_t k = 0; k < K; k++) {
                ntt::apply(ntt::fixed_shape<2>{}, [&](auto index) {
                    a0[index[0]] = a(m1 + index[0], k);
                });
                ntt::apply(ntt::fixed_shape<4>{}, [&](auto index) {
                    b0[index[0]] = b(n1 + index[0], k);
                });
                for (size_t m0 = 0; m0 < 2; m0++) {
                    for (size_t n0 = 0; n0 < 4; n0++) {
                        c0[m0][n0] = __riscv_vfmacc_vf_f32m1(c0[m0][n0], b0[n0],
                                                             a0[m0], vl);
                    }
                }
            }

            ntt::store(c(m1 + 0, n1 + 0), c0[0][0]);
            ntt::store(c(m1 + 0, n1 + 1), c0[0][1]);
            ntt::store(c(m1 + 0, n1 + 2), c0[0][2]);
            ntt::store(c(m1 + 0, n1 + 3), c0[0][3]);
            ntt::store(c(m1 + 1, n1 + 0), c0[1][0]);
            ntt::store(c(m1 + 1, n1 + 1), c0[1][1]);
            ntt::store(c(m1 + 1, n1 + 2), c0[1][2]);
            ntt::store(c(m1 + 1, n1 + 3), c0[1][3]);
        }
    }
}

template <bool Acc, class TC>
__attribute__((always_inline)) inline void
gemm_level5_l2(const ntt::vector<float, 8> *a, const float *b, TC &&c, size_t M,
               size_t K, size_t N) {
    constexpr size_t vl = NTT_VLEN / (sizeof(float) * 8);
    const ntt::vector<float, 8> *pa;
    const float *pb;

    for (size_t m1 = 0; m1 < M; m1 += 2) {
        for (size_t n1 = 0; n1 < N; n1 += 4) {
            fixed_vfloat32m1_t c0[2][4];
            fixed_vfloat32m1_t a0[2];
            float b0[4];
            ntt::apply(ntt::fixed_shape<2, 4>{}, [&](auto index) {
                c0[index[0]][index[1]] =
                    Acc ? (fixed_vfloat32m1_t)c(m1 + index[0], n1 + index[1])
                        : fixed_vfloat32m1_t{};
            });
            pa = a + m1 * 64;
            pb = b + n1 * 64;
            for (size_t k = 0; k < K; k++) {
                a0[0] = *pa++;
                a0[1] = *pa++;
                b0[0] = *pb++;
                b0[1] = *pb++;
                b0[2] = *pb++;
                b0[3] = *pb++;
                for (size_t m0 = 0; m0 < 2; m0++) {
                    for (size_t n0 = 0; n0 < 4; n0++) {
                        c0[m0][n0] = __riscv_vfmacc_vf_f32m1(c0[m0][n0], b0[n0],
                                                             a0[m0], vl);
                    }
                }
            }

            ntt::store(c(m1 + 0, n1 + 0), c0[0][0]);
            ntt::store(c(m1 + 0, n1 + 1), c0[0][1]);
            ntt::store(c(m1 + 0, n1 + 2), c0[0][2]);
            ntt::store(c(m1 + 0, n1 + 3), c0[0][3]);
            ntt::store(c(m1 + 1, n1 + 0), c0[1][0]);
            ntt::store(c(m1 + 1, n1 + 1), c0[1][1]);
            ntt::store(c(m1 + 1, n1 + 2), c0[1][2]);
            ntt::store(c(m1 + 1, n1 + 3), c0[1][3]);
        }
    }
}

template <size_t M, size_t K, size_t N>
void gemm_level5(ntt::fixed_tensor<ntt::vector<float, 8>, M, K> &a,
                 ntt::fixed_tensor<float, N, K> &b,
                 ntt::fixed_tensor<ntt::vector<float, 8>, M, N> &c) {
    constexpr size_t m2_tile = 32;
    constexpr size_t n2_tile = 128;
    constexpr size_t k2_tile = 256;

    alignas(4096) static ntt::fixed_tensor<ntt::vector<float, 8>, m2_tile / 2,
                                           k2_tile, 2>
        a2_packed;
    alignas(4096) static ntt::fixed_tensor<
        ntt::fixed_tensor<float, n2_tile / 4, k2_tile, 4>,
        ntt::ceil_div(N, n2_tile), ntt::ceil_div(K, k2_tile)>
        b2_packs;

    for (size_t m2 = 0; m2 < M; m2 += m2_tile) {
        auto real_m2 = std::min(m2_tile, M - m2);
        for (size_t k2 = 0; k2 < K; k2 += k2_tile) {
            auto real_k2 = std::min(k2_tile, K - k2);
            auto a2 = a.view(ntt::make_ranked_shape(m2, k2),
                             ntt::make_ranked_shape(real_m2, real_k2));
            for (size_t ma = 0; ma < real_m2; ma += 2) {
                for (size_t ka = 0; ka < real_k2; ka++) {
                    a2_packed(ma / 2, ka, 0) = a2(ma, ka);
                    a2_packed(ma / 2, ka, 1) = a2(ma + 1, ka);
                }
            }
            for (size_t n2 = 0; n2 < N; n2 += n2_tile) {
                auto real_n2 = std::min(n2_tile, N - n2);
                auto b2 = b.view(ntt::make_ranked_shape(n2, k2),
                                 ntt::make_ranked_shape(real_n2, real_k2));
                auto c2 = c.view(ntt::make_ranked_shape(m2, n2),
                                 ntt::make_ranked_shape(real_m2, real_n2));
                auto &b2_packed = b2_packs(n2 / n2_tile, k2 / k2_tile);
                if (m2 == 0) {
                    for (size_t nb = 0; nb < real_n2; nb += 4) {
                        for (size_t ka = 0; ka < real_k2; ka++) {
                            b2_packed(nb / 4, ka, 0) = b2(nb, ka);
                            b2_packed(nb / 4, ka, 1) = b2(nb + 1, ka);
                            b2_packed(nb / 4, ka, 2) = b2(nb + 2, ka);
                            b2_packed(nb / 4, ka, 3) = b2(nb + 3, ka);
                        }
                    }
                }
                if (k2 != 0) {
                    gemm_level5_l2<true>(a2_packed.elements().data(),
                                         b2_packed.elements().data(), c2,
                                         real_m2, real_k2, real_n2);
                } else {
                    gemm_level5_l2<false>(a2_packed.elements().data(),
                                          b2_packed.elements().data(), c2,
                                          real_m2, real_k2, real_n2);
                }
            }
        }
    }
}

#if 0
template <bool Acc, class TA, class TB, class TC>
__attribute__((noinline)) void gemm_level6_l2(TA &a, TB &b, TC &c) {
    const auto M = a.shape()[0];
    const auto N = b.shape()[0];
    const auto K = a.shape()[1];

    constexpr size_t vl = NTT_VLEN / (sizeof(float) * 8);

    for (size_t m1 = 0; m1 < M; m1 += 2) {
        __builtin_prefetch(&c(m1 + 2, 0), 1, 0);
        __builtin_prefetch(&c(m1 + 3, 0), 1, 0);
        __builtin_prefetch(&a(m1 + 2, 0), 0, 0);
        __builtin_prefetch(&a(m1 + 3, 0), 0, 0);
        for (size_t n1 = 0; n1 < N; n1 += 4) {
            fixed_vfloat32m1_t c0[2][4];
            fixed_vfloat32m1_t a0[2];
            float b0[4];
            ntt::apply(ntt::fixed_shape<2, 4>{}, [&](auto index) {
                c0[index[0]][index[1]] =
                    Acc ? (fixed_vfloat32m1_t)c(m1 + index[0], n1 + index[1])
                        : fixed_vfloat32m1_t{};
            });
            __builtin_prefetch(&b(n1 + 4, 0), 0, 0);
            __builtin_prefetch(&b(n1 + 5, 0), 0, 0);
            __builtin_prefetch(&b(n1 + 6, 0), 0, 0);
            __builtin_prefetch(&b(n1 + 7, 0), 0, 0);
            for (size_t k = 0; k < K; k++) {
                ntt::apply(ntt::fixed_shape<2>{}, [&](auto index) {
                    a0[index[0]] = a(m1 + index[0], k);
                });
                ntt::apply(ntt::fixed_shape<4>{}, [&](auto index) {
                    b0[index[0]] = b(n1 + index[0], k);
                });
                for (size_t m0 = 0; m0 < 2; m0++) {
                    for (size_t n0 = 0; n0 < 4; n0++) {
                        c0[m0][n0] = __riscv_vfmacc_vf_f32m1(c0[m0][n0], b0[n0],
                                                             a0[m0], vl);
                    }
                }
            }

            ntt::store(c(m1 + 0, n1 + 0), c0[0][0]);
            ntt::store(c(m1 + 0, n1 + 1), c0[0][1]);
            ntt::store(c(m1 + 0, n1 + 2), c0[0][2]);
            ntt::store(c(m1 + 0, n1 + 3), c0[0][3]);
            ntt::store(c(m1 + 1, n1 + 0), c0[1][0]);
            ntt::store(c(m1 + 1, n1 + 1), c0[1][1]);
            ntt::store(c(m1 + 1, n1 + 2), c0[1][2]);
            ntt::store(c(m1 + 1, n1 + 3), c0[1][3]);
        }
    }
}

template <size_t M, size_t K, size_t N>
void gemm_level6(ntt::fixed_tensor<ntt::vector<float, 8>, M, K> &a,
                 ntt::fixed_tensor<float, N, K> &b,
                 ntt::fixed_tensor<ntt::vector<float, 8>, M, N> &c) {
    constexpr size_t m2_tile = 8;
    constexpr size_t n2_tile = 32;
    constexpr size_t k2_tile = 64;

    for (size_t m2 = 0; m2 < M; m2 += m2_tile) {
        auto real_m2 = std::min(m2_tile, M - m2);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
        __builtin_prefetch(&c(m2 + m2_tile, 0), 1, 1);
        __builtin_prefetch(&a(m2 + m2_tile, 0), 0, 1);
#pragma GCC diagnostic pop
        for (size_t n2 = 0; n2 < N; n2 += n2_tile) {
            auto real_n2 = std::min(n2_tile, N - n2);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
            __builtin_prefetch(&b(n2 + n2_tile, 0), 0, 1);
#pragma GCC diagnostic pop
            for (size_t k2 = 0; k2 < K; k2 += k2_tile) {
                auto real_k2 = std::min(k2_tile, K - k2);
                auto a2 = a.view(ntt::make_ranked_shape(m2, k2),
                                 ntt::make_ranked_shape(real_m2, real_k2));
                auto b2 = b.view(ntt::make_ranked_shape(n2, k2),
                                 ntt::make_ranked_shape(real_n2, real_k2));
                auto c2 = c.view(ntt::make_ranked_shape(m2, n2),
                                 ntt::make_ranked_shape(real_m2, real_n2));
                if (k2 != 0) {
                    gemm_level6_l2<true>(a2, b2, c2);
                } else {
                    gemm_level6_l2<false>(a2, b2, c2);
                }
            }
        }
    }
}
#endif

template <bool Acc, class TC>
__attribute__((always_inline)) inline void
gemm_level6_l2(const ntt::vector<float, 8> *a, const float *b, TC &&c, size_t M,
               size_t K, size_t N) {
    for (size_t m1 = 0; m1 < M; m1 += 2) {
        for (size_t n1 = 0; n1 < N; n1 += 4) {
            register const fixed_vfloat32m1_t *a0_p asm("t0");
            register const float *b0_p asm("t1");

            register fixed_vfloat32m1_t c0_0_0 asm("v2");
            register fixed_vfloat32m1_t c0_0_1 asm("v3");
            register fixed_vfloat32m1_t c0_0_2 asm("v4");
            register fixed_vfloat32m1_t c0_0_3 asm("v5");
            register fixed_vfloat32m1_t c0_1_0 asm("v6");
            register fixed_vfloat32m1_t c0_1_1 asm("v7");
            register fixed_vfloat32m1_t c0_1_2 asm("v8");
            register fixed_vfloat32m1_t c0_1_3 asm("v9");

            if constexpr (Acc) {
                c0_0_0 = c(m1 + 0, n1 + 0);
                c0_0_1 = c(m1 + 0, n1 + 1);
                c0_0_2 = c(m1 + 0, n1 + 2);
                c0_0_3 = c(m1 + 0, n1 + 3);
                c0_1_0 = c(m1 + 1, n1 + 0);
                c0_1_1 = c(m1 + 1, n1 + 1);
                c0_1_2 = c(m1 + 1, n1 + 2);
                c0_1_3 = c(m1 + 1, n1 + 3);
            } else {
                c0_0_0 = fixed_vfloat32m1_t{};
                c0_0_1 = fixed_vfloat32m1_t{};
                c0_0_2 = fixed_vfloat32m1_t{};
                c0_0_3 = fixed_vfloat32m1_t{};
                c0_1_0 = fixed_vfloat32m1_t{};
                c0_1_1 = fixed_vfloat32m1_t{};
                c0_1_2 = fixed_vfloat32m1_t{};
                c0_1_3 = fixed_vfloat32m1_t{};
            }

            a0_p = (const fixed_vfloat32m1_t *)a + m1 * 64;
            b0_p = b + n1 * 64;
            for (size_t k = 0; k < K; k++) {
                asm volatile("vl1re32.v v0, (%[a0_p])\n"
                             "addi      %[a0_p], %[a0_p], 8 * 4\n"
                             "vl1re32.v	v1, (%[a0_p])\n"
                             "addi      %[a0_p], %[a0_p], 8 * 4\n"
                             "flw	    ft0, 0(%[b0_p])\n"
                             "flw	    ft1, 4(%[b0_p])\n"
                             "flw	    ft2, 8(%[b0_p])\n"
                             "flw	    ft3, 12(%[b0_p])\n"
                             "addi      %[b0_p], %[b0_p], 4 * 4\n"
                             "vfmacc.vf %[c0_0_0], ft0, v0\n"
                             "vfmacc.vf %[c0_0_1], ft1, v0\n"
                             "vfmacc.vf %[c0_0_2], ft2, v0\n"
                             "vfmacc.vf %[c0_0_3], ft3, v0\n"
                             "vfmacc.vf %[c0_1_0], ft0, v1\n"
                             "vfmacc.vf %[c0_1_1], ft1, v1\n"
                             "vfmacc.vf %[c0_1_2], ft2, v1\n"
                             "vfmacc.vf %[c0_1_3], ft3, v1\n"
                             : [a0_p] "+r"(a0_p), [b0_p] "+r"(b0_p),
                               [c0_0_0] "+vr"(c0_0_0), [c0_0_1] "+vr"(c0_0_1),
                               [c0_0_2] "+vr"(c0_0_2), [c0_0_3] "+vr"(c0_0_3),
                               [c0_1_0] "+vr"(c0_1_0), [c0_1_1] "+vr"(c0_1_1),
                               [c0_1_2] "+vr"(c0_1_2), [c0_1_3] "+vr"(c0_1_3)
                             :
                             : // Clobbers.
                             "cc", "memory",
                             // We use these Vector registers.
                             "v0", "v1",
                             // We use these general-purpose registers.
                             "ft0", "ft1", "ft2", "ft3");
            }

            ntt::store(c(m1 + 0, n1 + 0), c0_0_0);
            ntt::store(c(m1 + 0, n1 + 1), c0_0_1);
            ntt::store(c(m1 + 0, n1 + 2), c0_0_2);
            ntt::store(c(m1 + 0, n1 + 3), c0_0_3);
            ntt::store(c(m1 + 1, n1 + 0), c0_1_0);
            ntt::store(c(m1 + 1, n1 + 1), c0_1_1);
            ntt::store(c(m1 + 1, n1 + 2), c0_1_2);
            ntt::store(c(m1 + 1, n1 + 3), c0_1_3);
        }
    }
}

template <size_t M, size_t K, size_t N>
void gemm_level6(ntt::fixed_tensor<ntt::vector<float, 8>, M, K> &a,
                 ntt::fixed_tensor<float, N, K> &b,
                 ntt::fixed_tensor<ntt::vector<float, 8>, M, N> &c) {
    constexpr size_t m2_tile = 8;
    constexpr size_t n2_tile = 32;
    constexpr size_t k2_tile = 64;

    alignas(4096) static ntt::fixed_tensor<ntt::vector<float, 8>, m2_tile / 2,
                                           k2_tile, 2>
        a2_packed;
    alignas(4096) static ntt::fixed_tensor<
        ntt::fixed_tensor<float, n2_tile / 4, k2_tile, 4>,
        ntt::ceil_div(N, n2_tile), ntt::ceil_div(K, k2_tile)>
        b2_packs;

    for (size_t m2 = 0; m2 < M; m2 += m2_tile) {
        auto real_m2 = std::min(m2_tile, M - m2);
        for (size_t k2 = 0; k2 < K; k2 += k2_tile) {
            auto real_k2 = std::min(k2_tile, K - k2);
            auto a2 = a.view(ntt::make_ranked_shape(m2, k2),
                             ntt::make_ranked_shape(real_m2, real_k2));
            for (size_t ma = 0; ma < real_m2; ma += 2) {
                for (size_t ka = 0; ka < real_k2; ka++) {
                    a2_packed(ma / 2, ka, 0) = a2(ma, ka);
                    a2_packed(ma / 2, ka, 1) = a2(ma + 1, ka);
                }
            }
            for (size_t n2 = 0; n2 < N; n2 += n2_tile) {
                auto real_n2 = std::min(n2_tile, N - n2);
                auto b2 = b.view(ntt::make_ranked_shape(n2, k2),
                                 ntt::make_ranked_shape(real_n2, real_k2));
                auto c2 = c.view(ntt::make_ranked_shape(m2, n2),
                                 ntt::make_ranked_shape(real_m2, real_n2));
                auto &b2_packed = b2_packs(n2 / n2_tile, k2 / k2_tile);
                if (m2 == 0) {
                    for (size_t nb = 0; nb < real_n2; nb += 4) {
                        for (size_t ka = 0; ka < real_k2; ka++) {
                            b2_packed(nb / 4, ka, 0) = b2(nb, ka);
                            b2_packed(nb / 4, ka, 1) = b2(nb + 1, ka);
                            b2_packed(nb / 4, ka, 2) = b2(nb + 2, ka);
                            b2_packed(nb / 4, ka, 3) = b2(nb + 3, ka);
                        }
                    }
                }
                if (k2 != 0) {
                    gemm_level6_l2<true>(a2_packed.elements().data(),
                                         b2_packed.elements().data(), c2,
                                         real_m2, real_k2, real_n2);
                } else {
                    gemm_level6_l2<false>(a2_packed.elements().data(),
                                          b2_packed.elements().data(), c2,
                                          real_m2, real_k2, real_n2);
                }
            }
        }
    }
}

template <bool Acc, class TC>
__attribute__((always_inline)) inline void
gemm_level7_l2(const ntt::vector<float, 8> *a, const float *b, TC &&c, size_t M,
               size_t K, size_t N) {
#ifdef __riscv_vector
    // Enable RVV
    asm volatile("vsetivli	zero,8,e32,m1,ta,ma");
#endif
    for (size_t m1 = 0; m1 < M; m1 += 2) {
        for (size_t n1 = 0; n1 < N; n1 += 8) {
            register const fixed_vfloat32m1_t *a0_p asm("t0");
            register const float *b0_p asm("t1");

            register fixed_vfloat32m1_t c0_0_0 asm("v4");
            register fixed_vfloat32m1_t c0_0_1 asm("v5");
            register fixed_vfloat32m1_t c0_0_2 asm("v6");
            register fixed_vfloat32m1_t c0_0_3 asm("v7");
            register fixed_vfloat32m1_t c0_0_4 asm("v8");
            register fixed_vfloat32m1_t c0_0_5 asm("v9");
            register fixed_vfloat32m1_t c0_0_6 asm("v10");
            register fixed_vfloat32m1_t c0_0_7 asm("v11");
            register fixed_vfloat32m1_t c0_1_0 asm("v12");
            register fixed_vfloat32m1_t c0_1_1 asm("v13");
            register fixed_vfloat32m1_t c0_1_2 asm("v14");
            register fixed_vfloat32m1_t c0_1_3 asm("v15");
            register fixed_vfloat32m1_t c0_1_4 asm("v16");
            register fixed_vfloat32m1_t c0_1_5 asm("v17");
            register fixed_vfloat32m1_t c0_1_6 asm("v18");
            register fixed_vfloat32m1_t c0_1_7 asm("v19");

            if constexpr (Acc) {
                c0_0_0 = c(m1 + 0, n1 + 0);
                c0_0_1 = c(m1 + 0, n1 + 1);
                c0_0_2 = c(m1 + 0, n1 + 2);
                c0_0_3 = c(m1 + 0, n1 + 3);
                c0_0_4 = c(m1 + 0, n1 + 4);
                c0_0_5 = c(m1 + 0, n1 + 5);
                c0_0_6 = c(m1 + 0, n1 + 6);
                c0_0_7 = c(m1 + 0, n1 + 7);
                c0_1_0 = c(m1 + 1, n1 + 0);
                c0_1_1 = c(m1 + 1, n1 + 1);
                c0_1_2 = c(m1 + 1, n1 + 2);
                c0_1_3 = c(m1 + 1, n1 + 3);
                c0_1_4 = c(m1 + 1, n1 + 4);
                c0_1_5 = c(m1 + 1, n1 + 5);
                c0_1_6 = c(m1 + 1, n1 + 6);
                c0_1_7 = c(m1 + 1, n1 + 7);
            } else {
                c0_0_0 = fixed_vfloat32m1_t{};
                c0_0_1 = fixed_vfloat32m1_t{};
                c0_0_2 = fixed_vfloat32m1_t{};
                c0_0_3 = fixed_vfloat32m1_t{};
                c0_0_4 = fixed_vfloat32m1_t{};
                c0_0_5 = fixed_vfloat32m1_t{};
                c0_0_6 = fixed_vfloat32m1_t{};
                c0_0_7 = fixed_vfloat32m1_t{};
                c0_1_0 = fixed_vfloat32m1_t{};
                c0_1_1 = fixed_vfloat32m1_t{};
                c0_1_2 = fixed_vfloat32m1_t{};
                c0_1_3 = fixed_vfloat32m1_t{};
                c0_1_4 = fixed_vfloat32m1_t{};
                c0_1_5 = fixed_vfloat32m1_t{};
                c0_1_6 = fixed_vfloat32m1_t{};
                c0_1_7 = fixed_vfloat32m1_t{};
            }

            a0_p = (const fixed_vfloat32m1_t *)a + m1 * 64;
            b0_p = b + n1 * 64;

            // 1. Preload
            asm volatile("vl1re32.v v0, (%[a0_p])\n"
                         "addi      %[a0_p], %[a0_p], 8 * 4\n"
                         "vl1re32.v	v1, (%[a0_p])\n"
                         "addi      %[a0_p], %[a0_p], 8 * 4\n"
                         "flw	    ft0, 0(%[b0_p])\n"
                         "flw	    ft1, 4(%[b0_p])\n"
                         "flw	    ft2, 8(%[b0_p])\n"
                         "flw	    ft3, 12(%[b0_p])\n"
                         "flw	    ft4, 16(%[b0_p])\n"
                         "flw	    ft5, 20(%[b0_p])\n"
                         "flw	    ft6, 24(%[b0_p])\n"
                         "flw	    ft7, 28(%[b0_p])\n"
                         "addi      %[b0_p], %[b0_p], 8 * 4\n"
                         : [a0_p] "+r"(a0_p), [b0_p] "+r"(b0_p)
                         :
                         : // Clobbers.
                         "cc", "memory",
                         // We use these Vector registers.
                         "v0", "v1",
                         // We use these general-purpose registers.
                         "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6",
                         "ft7");

// 2. Pipelined
#define GEMM_PING                                                              \
    asm volatile("vfmacc.vf %[c0_0_0], ft0, v0\n"                              \
                 "vfmacc.vf %[c0_0_1], ft1, v0\n"                              \
                 "vl1re32.v v2, (%[a0_p])\n"                                   \
                 "addi      %[a0_p], %[a0_p], 8 * 4\n"                         \
                 "vfmacc.vf %[c0_0_2], ft2, v0\n"                              \
                 "vl1re32.v v3, (%[a0_p])\n"                                   \
                 "addi      %[a0_p], %[a0_p], 8 * 4\n"                         \
                 "vfmacc.vf %[c0_0_3], ft3, v0\n"                              \
                 "flw	    fa0, 0(%[b0_p])\n"                                   \
                 "vfmacc.vf %[c0_0_4], ft4, v0\n"                              \
                 "flw	    fa1, 4(%[b0_p])\n"                                   \
                 "vfmacc.vf %[c0_0_5], ft5, v0\n"                              \
                 "flw	    fa2, 8(%[b0_p])\n"                                   \
                 "vfmacc.vf %[c0_0_6], ft6, v0\n"                              \
                 "flw	    fa3, 12(%[b0_p])\n"                                  \
                 "vfmacc.vf %[c0_0_7], ft7, v0\n"                              \
                 : [a0_p] "+r"(a0_p), [b0_p] "+r"(b0_p),                       \
                   [c0_0_0] "+vr"(c0_0_0), [c0_0_1] "+vr"(c0_0_1),             \
                   [c0_0_2] "+vr"(c0_0_2), [c0_0_3] "+vr"(c0_0_3),             \
                   [c0_0_4] "+vr"(c0_0_4), [c0_0_5] "+vr"(c0_0_5),             \
                   [c0_0_6] "+vr"(c0_0_6), [c0_0_7] "+vr"(c0_0_7)::"cc",       \
                   "memory", "v0", "v1", "v2", "v3", "ft0", "ft1", "ft2",      \
                   "ft3", "ft4", "ft5", "ft6", "ft7", "fa0", "fa1", "fa2",     \
                   "fa3");                                                     \
                                                                               \
    asm volatile(                                                              \
        "flw	    fa4, 16(%[b0_p])\n"                                           \
        "vfmacc.vf %[c0_1_0], ft0, v1\n"                                       \
        "flw	    fa5, 20(%[b0_p])\n"                                           \
        "vfmacc.vf %[c0_1_1], ft1, v1\n"                                       \
        "flw	    fa6, 24(%[b0_p])\n"                                           \
        "vfmacc.vf %[c0_1_2], ft2, v1\n"                                       \
        "flw	    fa7, 28(%[b0_p])\n"                                           \
        "vfmacc.vf %[c0_1_3], ft3, v1\n"                                       \
        "vfmacc.vf %[c0_1_4], ft4, v1\n"                                       \
        "vfmacc.vf %[c0_1_5], ft5, v1\n"                                       \
        "vfmacc.vf %[c0_1_6], ft6, v1\n"                                       \
        "vfmacc.vf %[c0_1_7], ft7, v1\n"                                       \
        : [b0_p] "+r"(b0_p), [c0_1_0] "+vr"(c0_1_0), [c0_1_1] "+vr"(c0_1_1),   \
          [c0_1_2] "+vr"(c0_1_2), [c0_1_3] "+vr"(c0_1_3),                      \
          [c0_1_4] "+vr"(c0_1_4), [c0_1_5] "+vr"(c0_1_5),                      \
          [c0_1_6] "+vr"(c0_1_6), [c0_1_7] "+vr"(c0_1_7)::"cc", "memory",      \
          "v0", "v1", "v2", "v3", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5",    \
          "ft6", "ft7", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6",       \
          "fa7");

#define GEMM_PONG                                                              \
    asm volatile("vfmacc.vf %[c0_0_0], fa0, v2\n"                              \
                 "vl1re32.v v0, (%[a0_p])\n"                                   \
                 "addi      %[a0_p], %[a0_p], 8 * 4\n"                         \
                 "vfmacc.vf %[c0_0_1], fa1, v2\n"                              \
                 "vl1re32.v v1, (%[a0_p])\n"                                   \
                 "addi      %[a0_p], %[a0_p], 8 * 4\n"                         \
                 "vfmacc.vf %[c0_0_2], fa2, v2\n"                              \
                 "flw	    ft0, 32(%[b0_p])\n"                                  \
                 "vfmacc.vf %[c0_0_3], fa3, v2\n"                              \
                 "flw	    ft1, 36(%[b0_p])\n"                                  \
                 "vfmacc.vf %[c0_0_4], fa4, v2\n"                              \
                 "vfmacc.vf %[c0_0_5], fa5, v2\n"                              \
                 "flw	    ft2, 40(%[b0_p])\n"                                  \
                 "vfmacc.vf %[c0_0_6], fa6, v2\n"                              \
                 "vfmacc.vf %[c0_0_7], fa7, v2\n"                              \
                 "flw	    ft3, 44(%[b0_p])\n"                                  \
                 : [a0_p] "+r"(a0_p), [b0_p] "+r"(b0_p),                       \
                   [c0_0_0] "+vr"(c0_0_0), [c0_0_1] "+vr"(c0_0_1),             \
                   [c0_0_2] "+vr"(c0_0_2), [c0_0_3] "+vr"(c0_0_3),             \
                   [c0_0_4] "+vr"(c0_0_4), [c0_0_5] "+vr"(c0_0_5),             \
                   [c0_0_6] "+vr"(c0_0_6), [c0_0_7] "+vr"(c0_0_7)::"cc",       \
                   "memory", "v0", "v1", "v2", "v3", "ft0", "ft1", "ft2",      \
                   "ft3", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6",     \
                   "fa7");                                                     \
                                                                               \
    asm volatile(                                                              \
        "vfmacc.vf %[c0_1_0], fa0, v3\n"                                       \
        "flw	    ft4, 48(%[b0_p])\n"                                           \
        "vfmacc.vf %[c0_1_1], fa1, v3\n"                                       \
        "flw	    ft5, 52(%[b0_p])\n"                                           \
        "vfmacc.vf %[c0_1_2], fa2, v3\n"                                       \
        "flw	    ft6, 56(%[b0_p])\n"                                           \
        "vfmacc.vf %[c0_1_3], fa3, v3\n"                                       \
        "flw	    ft7, 60(%[b0_p])\n"                                           \
        "addi      %[b0_p], %[b0_p], 16 * 4\n"                                 \
        "vfmacc.vf %[c0_1_4], fa4, v3\n"                                       \
        "vfmacc.vf %[c0_1_5], fa5, v3\n"                                       \
        "vfmacc.vf %[c0_1_6], fa6, v3\n"                                       \
        "vfmacc.vf %[c0_1_7], fa7, v3\n"                                       \
        : [b0_p] "+r"(b0_p), [c0_1_0] "+vr"(c0_1_0), [c0_1_1] "+vr"(c0_1_1),   \
          [c0_1_2] "+vr"(c0_1_2), [c0_1_3] "+vr"(c0_1_3),                      \
          [c0_1_4] "+vr"(c0_1_4), [c0_1_5] "+vr"(c0_1_5),                      \
          [c0_1_6] "+vr"(c0_1_6), [c0_1_7] "+vr"(c0_1_7)::"cc", "memory",      \
          "v0", "v1", "v2", "v3", "ft0", "ft1", "ft2", "ft3", "ft4", "ft5",    \
          "ft6", "ft7", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6",       \
          "fa7");

            const size_t pipeline_count = (K - 1) / 2;
            for (size_t k1 = 0; k1 < pipeline_count; k1++) {
                GEMM_PING
                GEMM_PONG
            }

            // 3. Tail
            if (K % 2 == 0) {
                GEMM_PING
                asm volatile("vfmacc.vf %[c0_0_0], fa0, v2\n"
                             "vfmacc.vf %[c0_0_1], fa1, v2\n"
                             "vfmacc.vf %[c0_0_2], fa2, v2\n"
                             "vfmacc.vf %[c0_0_3], fa3, v2\n"
                             "vfmacc.vf %[c0_0_4], fa4, v2\n"
                             "vfmacc.vf %[c0_0_5], fa5, v2\n"
                             "vfmacc.vf %[c0_0_6], fa6, v2\n"
                             "vfmacc.vf %[c0_0_7], fa7, v2\n"
                             : [c0_0_0] "+vr"(c0_0_0), [c0_0_1] "+vr"(c0_0_1),
                               [c0_0_2] "+vr"(c0_0_2), [c0_0_3] "+vr"(c0_0_3),
                               [c0_0_4] "+vr"(c0_0_4), [c0_0_5] "+vr"(c0_0_5),
                               [c0_0_6] "+vr"(c0_0_6), [c0_0_7] "+vr"(c0_0_7)
                             :
                             : "cc", "memory", "v2", "fa0", "fa1", "fa2", "fa3",
                               "fa4", "fa5", "fa6", "fa7");
                asm volatile("vfmacc.vf %[c0_1_0], fa0, v3\n"
                             "vfmacc.vf %[c0_1_1], fa1, v3\n"
                             "vfmacc.vf %[c0_1_2], fa2, v3\n"
                             "vfmacc.vf %[c0_1_3], fa3, v3\n"
                             "vfmacc.vf %[c0_1_4], fa4, v3\n"
                             "vfmacc.vf %[c0_1_5], fa5, v3\n"
                             "vfmacc.vf %[c0_1_6], fa6, v3\n"
                             "vfmacc.vf %[c0_1_7], fa7, v3\n"
                             : [c0_1_0] "+vr"(c0_1_0), [c0_1_1] "+vr"(c0_1_1),
                               [c0_1_2] "+vr"(c0_1_2), [c0_1_3] "+vr"(c0_1_3),
                               [c0_1_4] "+vr"(c0_1_4), [c0_1_5] "+vr"(c0_1_5),
                               [c0_1_6] "+vr"(c0_1_6), [c0_1_7] "+vr"(c0_1_7)
                             :
                             : "cc", "memory", "v3", "fa0", "fa1", "fa2", "fa3",
                               "fa4", "fa5", "fa6", "fa7");
            } else {
                asm volatile("vfmacc.vf %[c0_0_0], ft0, v0\n"
                             "vfmacc.vf %[c0_0_1], ft1, v0\n"
                             "vfmacc.vf %[c0_0_2], ft2, v0\n"
                             "vfmacc.vf %[c0_0_3], ft3, v0\n"
                             "vfmacc.vf %[c0_0_4], ft4, v0\n"
                             "vfmacc.vf %[c0_0_5], ft5, v0\n"
                             "vfmacc.vf %[c0_0_6], ft6, v0\n"
                             "vfmacc.vf %[c0_0_7], ft7, v0\n"
                             : [c0_0_0] "+vr"(c0_0_0), [c0_0_1] "+vr"(c0_0_1),
                               [c0_0_2] "+vr"(c0_0_2), [c0_0_3] "+vr"(c0_0_3),
                               [c0_0_4] "+vr"(c0_0_4), [c0_0_5] "+vr"(c0_0_5),
                               [c0_0_6] "+vr"(c0_0_6), [c0_0_7] "+vr"(c0_0_7)
                             :
                             : "cc", "memory", "v1", "ft0", "ft1", "ft2", "ft3",
                               "ft4", "ft5", "ft6", "ft7");
                asm volatile("vfmacc.vf %[c0_1_0], ft0, v1\n"
                             "vfmacc.vf %[c0_1_1], ft1, v1\n"
                             "vfmacc.vf %[c0_1_2], ft2, v1\n"
                             "vfmacc.vf %[c0_1_3], ft3, v1\n"
                             "vfmacc.vf %[c0_1_4], ft4, v1\n"
                             "vfmacc.vf %[c0_1_5], ft5, v1\n"
                             "vfmacc.vf %[c0_1_6], ft6, v1\n"
                             "vfmacc.vf %[c0_1_7], ft7, v1\n"
                             : [c0_1_0] "+vr"(c0_1_0), [c0_1_1] "+vr"(c0_1_1),
                               [c0_1_2] "+vr"(c0_1_2), [c0_1_3] "+vr"(c0_1_3),
                               [c0_1_4] "+vr"(c0_1_4), [c0_1_5] "+vr"(c0_1_5),
                               [c0_1_6] "+vr"(c0_1_6), [c0_1_7] "+vr"(c0_1_7)
                             :
                             : "cc", "memory", "v1", "ft0", "ft1", "ft2", "ft3",
                               "ft4", "ft5", "ft6", "ft7");
            }

            ntt::store(c(m1 + 0, n1 + 0), c0_0_0);
            ntt::store(c(m1 + 0, n1 + 1), c0_0_1);
            ntt::store(c(m1 + 0, n1 + 2), c0_0_2);
            ntt::store(c(m1 + 0, n1 + 3), c0_0_3);
            ntt::store(c(m1 + 0, n1 + 4), c0_0_4);
            ntt::store(c(m1 + 0, n1 + 5), c0_0_5);
            ntt::store(c(m1 + 0, n1 + 6), c0_0_6);
            ntt::store(c(m1 + 0, n1 + 7), c0_0_7);
            ntt::store(c(m1 + 1, n1 + 0), c0_1_0);
            ntt::store(c(m1 + 1, n1 + 1), c0_1_1);
            ntt::store(c(m1 + 1, n1 + 2), c0_1_2);
            ntt::store(c(m1 + 1, n1 + 3), c0_1_3);
            ntt::store(c(m1 + 1, n1 + 4), c0_1_4);
            ntt::store(c(m1 + 1, n1 + 5), c0_1_5);
            ntt::store(c(m1 + 1, n1 + 6), c0_1_6);
            ntt::store(c(m1 + 1, n1 + 7), c0_1_7);
        }
    }
}

template <size_t M, size_t K, size_t N>
void gemm_level7(ntt::fixed_tensor<ntt::vector<float, 8>, M, K> &a,
                 ntt::fixed_tensor<float, N, K> &b,
                 ntt::fixed_tensor<ntt::vector<float, 8>, M, N> &c) {
    constexpr size_t m2_tile = 8;
    constexpr size_t n2_tile = 32;
    constexpr size_t k2_tile = 64;

    alignas(4096) static ntt::fixed_tensor<ntt::vector<float, 8>, m2_tile / 2,
                                           k2_tile, 2>
        a2_packed;
    alignas(4096) static ntt::fixed_tensor<
        ntt::fixed_tensor<float, n2_tile / 8, k2_tile, 8>,
        ntt::ceil_div(N, n2_tile), ntt::ceil_div(K, k2_tile)>
        b2_packs;

    for (size_t m2 = 0; m2 < M; m2 += m2_tile) {
        auto real_m2 = std::min(m2_tile, M - m2);
        for (size_t k2 = 0; k2 < K; k2 += k2_tile) {
            auto real_k2 = std::min(k2_tile, K - k2);
            auto a2 = a.view(ntt::make_ranked_shape(m2, k2),
                             ntt::make_ranked_shape(real_m2, real_k2));
            for (size_t ma = 0; ma < real_m2; ma += 2) {
                for (size_t ka = 0; ka < real_k2; ka++) {
                    a2_packed(ma / 2, ka, 0) = a2(ma, ka);
                    a2_packed(ma / 2, ka, 1) = a2(ma + 1, ka);
                }
            }
            for (size_t n2 = 0; n2 < N; n2 += n2_tile) {
                auto real_n2 = std::min(n2_tile, N - n2);
                auto b2 = b.view(ntt::make_ranked_shape(n2, k2),
                                 ntt::make_ranked_shape(real_n2, real_k2));
                auto c2 = c.view(ntt::make_ranked_shape(m2, n2),
                                 ntt::make_ranked_shape(real_m2, real_n2));
                auto &b2_packed = b2_packs(n2 / n2_tile, k2 / k2_tile);
                if (m2 == 0) {
                    for (size_t nb = 0; nb < real_n2; nb += 4) {
                        for (size_t ka = 0; ka < real_k2; ka++) {
                            b2_packed(nb / 8, ka, 0) = b2(nb, ka);
                            b2_packed(nb / 8, ka, 1) = b2(nb + 1, ka);
                            b2_packed(nb / 8, ka, 2) = b2(nb + 2, ka);
                            b2_packed(nb / 8, ka, 3) = b2(nb + 3, ka);
                            b2_packed(nb / 8, ka, 4) = b2(nb + 4, ka);
                            b2_packed(nb / 8, ka, 5) = b2(nb + 5, ka);
                            b2_packed(nb / 8, ka, 6) = b2(nb + 6, ka);
                            b2_packed(nb / 8, ka, 7) = b2(nb + 7, ka);
                        }
                    }
                }
                if (k2 != 0) {
                    gemm_level7_l2<true>(a2_packed.elements().data(),
                                         b2_packed.elements().data(), c2,
                                         real_m2, real_k2, real_n2);
                } else {
                    gemm_level7_l2<false>(a2_packed.elements().data(),
                                          b2_packed.elements().data(), c2,
                                          real_m2, real_k2, real_n2);
                }
            }
        }
    }
}
#endif

// pack M
template <size_t M, size_t K, size_t N> void benchmark_ntt_matmul_pack_M() {
    constexpr size_t warmup_num = 10;
    constexpr size_t run_num = M < 128 ? 30 : 10;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    static ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
    static ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
    static ntt::tensor<float, ntt::fixed_shape<M, N>> tc1, tc2;
    NttTest::init_tensor(ta, -2.f, 2.f);
    NttTest::init_tensor(tb, -2.f, 2.f);
    alignas(32) static ntt::tensor<ntt::vector<float, P>,
                                   ntt::fixed_shape<M / P, K>>
        pa;
    alignas(32) static ntt::tensor<ntt::vector<float, P>,
                                   ntt::fixed_shape<M / P, N>>
        pc;
    ntt::pack<0>(ta, pa);

    gemm_naive(ta, tb, tc1);
    GEMM_NAME(pa, tb, pc);
    ntt::unpack<0>(pc, tc2);
    if (!NttTest::compare_tensor(tc1, tc2)) {
        std::cerr << "Test failed" << std::endl;
        return;
    }

    for (size_t i = 0; i < warmup_num; i++)
        GEMM_NAME(pa, tb, pc);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < run_num; i++) {
        GEMM_NAME(pa, tb, pc);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    asm volatile("" ::"g"(pc));

    auto ops = M * N * K * 2;
    // auto t = get_time(&start, &end) / run_num;
    auto t =
        std::chrono::duration<double, std::ratio<1>>(stop - start).count() /
        run_num;
    std::cout << (__FUNCTION__ + std::strlen("benchmark_ntt_matmul_pack_"))
              << std::setprecision(0) << std::fixed << ", M:" << M
              << ", K:" << K << ", N:" << N
              << ", GFLOPS:" << std::setprecision(1) << std::fixed
              << ops / t * 1e-9 << std::endl;
}

// pack M BT
template <size_t M, size_t K, size_t N> void benchmark_ntt_matmul_pack_M_BT() {
    constexpr size_t warmup_num = 1;
    constexpr size_t run_num = 1;
    constexpr size_t P = NTT_VLEN / (sizeof(float) * 8);

    alignas(64) static ntt::tensor<float, ntt::fixed_shape<M, K>> ta;
    alignas(64) static ntt::tensor<float, ntt::fixed_shape<K, N>> tb;
    alignas(64) static ntt::tensor<float, ntt::fixed_shape<N, K>> tbt;
    alignas(64) static ntt::tensor<float, ntt::fixed_shape<M, N>> tc1, tc2;
    NttTest::init_tensor(ta, -2.f, 2.f);
    NttTest::init_tensor(tb, -2.f, 2.f);
    alignas(64) static ntt::tensor<ntt::vector<float, P>,
                                   ntt::fixed_shape<M / P, K>>
        pa;
    alignas(64) static ntt::tensor<ntt::vector<float, P>,
                                   ntt::fixed_shape<M / P, N>>
        pc;
    ntt::pack<0>(ta, pa);
    ntt::transpose<ntt::fixed_shape<1, 0>>(tb, tbt);

    gemm_naive(ta, tb, tc1);
    GEMM_NAME(pa, tbt, pc);
    ntt::unpack<0>(pc, tc2);
#if 0
    if (!NttTest::compare_tensor(tc1, tc2)) {
        std::cerr << "Test failed" << std::endl;
        return;
    }
#endif

    for (size_t i = 0; i < warmup_num; i++)
        GEMM_NAME(pa, tbt, pc);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < run_num; i++) {
        GEMM_NAME(pa, tbt, pc);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    asm volatile("" ::"g"(pc));

    auto ops = M * N * K * 2;
    // auto t = get_time(&start, &end) / run_num;
    auto t =
        std::chrono::duration<double, std::ratio<1>>(stop - start).count() /
        run_num;
    std::cout << (__FUNCTION__ + std::strlen("benchmark_ntt_matmul_pack_"))
              << std::setprecision(0) << std::fixed << ", M:" << M
              << ", K:" << K << ", N:" << N
              << ", GFLOPS:" << std::setprecision(1) << std::fixed
              << ops / t * 1e-9 << std::endl;
}

#define BENCHMARK_NTT_MATMUL_1(MODE, M, K, N)                                  \
    benchmark_ntt_matmul_pack_##MODE<N * 1, K * 1, N * 1>();

#define BENCHMARK_NTT_MATMUL_2(MODE, N) BENCHMARK_NTT_MATMUL_1(MODE, N, N, N)

#if 1
#define BENCHMARK_NTT_MATMUL(MODE)                                             \
    BENCHMARK_NTT_MATMUL_2(MODE, 32)                                           \
    BENCHMARK_NTT_MATMUL_2(MODE, 64)                                           \
    BENCHMARK_NTT_MATMUL_2(MODE, 128)                                          \
    BENCHMARK_NTT_MATMUL_2(MODE, 256)                                          \
    BENCHMARK_NTT_MATMUL_2(MODE, 512)                                          \
    BENCHMARK_NTT_MATMUL_2(MODE, 1024)                                         \
    BENCHMARK_NTT_MATMUL_2(MODE, 2048)
#else
#define BENCHMARK_NTT_MATMUL(MODE) BENCHMARK_NTT_MATMUL_2(MODE, 512)
#endif

template <nncase::ntt::ukernels::mamtul_pack_kind PackKind>
void matmul_primitive_analysis() {
    switch (PackKind) {
    case ntt::ukernels::mamtul_pack_kind::no_pack:
        BENCHMARK_NTT_MATMUL(NONE);
        break;
    case ntt::ukernels::mamtul_pack_kind::pack_m:
#if GEMM_VERSION >= 4
        BENCHMARK_NTT_MATMUL(M_BT);
#else
        BENCHMARK_NTT_MATMUL(M);
#endif
        break;
    default:
        std::cout << "Invalid packing kind" << std::endl;
        break;
    }
}

int main() {
#if 0
    {
        const auto PackMode = nncase::ntt::ukernels::mamtul_pack_kind::no_pack;
        matmul_primitive_analysis<PackMode>();
    }
#else
    {
        const auto PackMode = nncase::ntt::ukernels::mamtul_pack_kind::pack_m;
        matmul_primitive_analysis<PackMode>();
    }
#endif

    return 0;
}