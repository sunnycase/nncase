/* Copyright 2019-2021 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "bfloat16.h"
#include "math.h"
#include "ntt/compiler_defs.h"
#include <bit>
#include <codecvt>
#include <cstdint>
#include <float.h>
#include <functional>
#include <limits>
#include <type_traits>

#ifdef __CUDA_ARCH__
#include <cuda_fp16.h>
#elif defined(__F16C__)
#include <immintrin.h>
#endif

namespace nncase {
#ifdef __CUDA_ARCH__
using native_half_t = __half;
#else
using native_half_t = _Float16;
#endif

struct fp16_from_raw_t {
    explicit fp16_from_raw_t() = default;
};

inline constexpr fp16_from_raw_t fp16_from_raw{};

struct half {
  private:
    static constexpr uint16_t ZERO_VALUE = 0;

    // this is quiet NaN, sNaN only used for send signal
    static constexpr uint16_t NAN_VALUE = 0x7e00;

  public:
    NTT_HOST_DEVICE constexpr half() noexcept = default;
    NTT_HOST_DEVICE constexpr half(native_half_t v) noexcept : value_(v) {}

    template <class T,
              class = std::enable_if_t<std::is_integral<T>::value ||
                                       std::is_floating_point<T>::value>>
    NTT_HOST_DEVICE constexpr explicit half(const T &v) noexcept
        : value_(round_to_half(v).value_) {}

    NTT_HOST_DEVICE static constexpr half round_to_half(float v) {
        if (std::is_constant_evaluated()) {
            return (native_half_t)v;
        } else {
#ifdef __CUDA_ARCH__
            return __float2half_rn(v);
#elif defined(__F16C__)
            // To avoid truncsfhf2
            return from_raw(_cvtss_sh(v, _MM_FROUND_NEARBYINT));
#else
            return (_Float16)v;
#endif
        }

        return (native_half_t)v;
    }

    NTT_HOST_DEVICE static constexpr half epsilon() noexcept { return from_raw(0x0800); }

    // Integer conversion constructors
    NTT_HOST_DEVICE constexpr explicit half(int x) noexcept
        : value_(round_to_half(float(x)).value_) {}

    NTT_HOST_DEVICE constexpr explicit half(int64_t x) noexcept
        : value_(round_to_half(float(x)).value_) {}

    NTT_HOST_DEVICE constexpr explicit half(uint32_t x) noexcept
        : value_(round_to_half(float(x)).value_) {}

    NTT_HOST_DEVICE constexpr explicit half(uint64_t x) noexcept
        : value_(round_to_half(double(x)).value_) {}

    // Floating point conversion constructors
    NTT_HOST_DEVICE constexpr explicit half(double x) noexcept
        : value_(round_to_half(float(x)).value_) {}

    // bfloat16 conversion constructor
    NTT_HOST_DEVICE constexpr explicit half(bfloat16 x) noexcept
        : value_(round_to_half(float(x)).value_) {}

    NTT_HOST_DEVICE constexpr half(fp16_from_raw_t, uint16_t value) noexcept
        : value_(std::bit_cast<native_half_t>(value)) {}

    NTT_HOST_DEVICE constexpr operator native_half_t() const noexcept { return value_; }
    NTT_HOST_DEVICE constexpr operator float() const noexcept {
        if (std::is_constant_evaluated()) {
            return (float)value_;
        } else {
#ifdef __CUDA_ARCH__
            return __half2float(value_);
#elif defined(__F16C__)
            // To avoid extendhfdf2
            return _cvtsh_ss(raw());
#else
            return (float)value_;
#endif
        }
    }

    NTT_HOST_DEVICE constexpr uint16_t raw() const noexcept {
        return std::bit_cast<uint16_t>(value_);
    }

    NTT_HOST_DEVICE static constexpr half from_raw(uint16_t v) noexcept {
        return half(nncase::fp16_from_raw, v);
    }

    // Type conversion operators
    NTT_HOST_DEVICE constexpr explicit operator double() const noexcept {
        return double(float(*this));
    }

    NTT_HOST_DEVICE constexpr explicit operator int8_t() const noexcept {
        return int(float(*this));
    }

    NTT_HOST_DEVICE constexpr explicit operator uint8_t() const noexcept {
        return int(float(*this));
    }

    NTT_HOST_DEVICE constexpr explicit operator int16_t() const noexcept {
        return int(float(*this));
    }

    NTT_HOST_DEVICE constexpr explicit operator uint16_t() const noexcept {
        return int(float(*this));
    }

    NTT_HOST_DEVICE constexpr explicit operator int() const noexcept {
        return int(float(*this));
    }

    NTT_HOST_DEVICE constexpr explicit operator int64_t() const noexcept {
        return int64_t(float(*this));
    }

    NTT_HOST_DEVICE constexpr explicit operator uint32_t() const noexcept {
        return uint32_t(float(*this));
    }

    NTT_HOST_DEVICE constexpr explicit operator uint64_t() const noexcept {
        return uint64_t(double(*this));
    }

    NTT_HOST_DEVICE constexpr explicit operator bool() const noexcept {
        return bool(std::bit_cast<uint16_t>(*this));
    }

    NTT_HOST_DEVICE static constexpr half highest() noexcept { return from_raw(0x7bff); }

    NTT_HOST_DEVICE static constexpr half min() noexcept { return from_raw(0x0400); }

    NTT_HOST_DEVICE static constexpr half lowest() noexcept { return from_raw(0xfbff); }

    NTT_HOST_DEVICE static constexpr half quiet_NaN() noexcept { return from_raw(0x7e00); }

    NTT_HOST_DEVICE static constexpr half signaling_NaN() noexcept {
        return from_raw(0x7d00);
    }

    NTT_HOST_DEVICE static constexpr half infinity() noexcept { return from_raw(0x7c00); }

    NTT_HOST_DEVICE constexpr bool zero() const noexcept {
        return (raw() & 0x7FFF) == ZERO_VALUE;
    }

    void operator=(const float &v) noexcept {
        value_ = (round_to_half(v).value_);
    }

  private:
    native_half_t value_;
};

#define DEFINE_FP16_BINARY_FP16RET(x)                                          \
    NTT_ALWAYS_INLINE NTT_HOST_DEVICE half operator x(half a, half b) noexcept { \
        return half::round_to_half(float(a) x float(b));                       \
    }

#define DEFINE_FP16_BINARY_BOOLRET(x)                                          \
    NTT_ALWAYS_INLINE NTT_HOST_DEVICE bool operator x(half a, half b) noexcept { \
        return float(a) x float(b);                                            \
    }

#define DEFINE_FP16_BINARY_FP32RET(x)                                          \
    NTT_ALWAYS_INLINE NTT_HOST_DEVICE bool operator x(half a, float b) noexcept { \
        return float(a) x b;                                                   \
    }

#define DEFINE_FP16_BINARY_INTRET(x)                                           \
    NTT_ALWAYS_INLINE NTT_HOST_DEVICE half operator x(half a, int b) noexcept { \
        return half::round_to_half(float(a) x b);                              \
    }

DEFINE_FP16_BINARY_FP32RET(<)
DEFINE_FP16_BINARY_INTRET(-)
DEFINE_FP16_BINARY_INTRET(+)
DEFINE_FP16_BINARY_INTRET(*)
DEFINE_FP16_BINARY_INTRET(/)

DEFINE_FP16_BINARY_FP16RET(+)
DEFINE_FP16_BINARY_FP16RET(-)
DEFINE_FP16_BINARY_FP16RET(*)
DEFINE_FP16_BINARY_FP16RET(/)
DEFINE_FP16_BINARY_BOOLRET(<)
DEFINE_FP16_BINARY_BOOLRET(<=)
DEFINE_FP16_BINARY_BOOLRET(>=)
DEFINE_FP16_BINARY_BOOLRET(>)

#define DEFINE_FP16_BINARY_SELF_MOD(x, op)                                     \
    NTT_ALWAYS_INLINE NTT_HOST_DEVICE half &operator x(half &a, half b) noexcept { \
        a = a op b;                                                            \
        return a;                                                              \
    }

DEFINE_FP16_BINARY_SELF_MOD(+=, +)
DEFINE_FP16_BINARY_SELF_MOD(-=, -)
DEFINE_FP16_BINARY_SELF_MOD(*=, *)
DEFINE_FP16_BINARY_SELF_MOD(/=, /)

NTT_ALWAYS_INLINE NTT_HOST_DEVICE half operator-(half a) noexcept {
    return half::round_to_half(-float(a));
}

NTT_ALWAYS_INLINE NTT_HOST_DEVICE bool operator==(const half &lhs, const half &rhs) noexcept {
    return lhs.raw() == rhs.raw();
}

NTT_ALWAYS_INLINE NTT_HOST_DEVICE bool operator!=(const half &lhs, const half &rhs) noexcept {
    return lhs.raw() != rhs.raw();
}

inline std::ostream &operator<<(std::ostream &os, const half &a) {
    os << std::to_string(float(a));
    return os;
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half nextafter(const half &from,
                                                 const half &to) {
    if (from.raw() == to.raw()) {
        return to;
    }

    const uint16_t from_raw = from.raw();
    const uint16_t to_raw = to.raw();

    const bool is_to_larger =
        (from_raw < to_raw) ^ ((from_raw ^ to_raw) & 0x8000);

    if (from.zero()) {
        return is_to_larger ? half::from_raw(0x0001)  // +0 -> +min_positive
                            : half::from_raw(0x8001); // +0 -> -max_negative
    }

    uint16_t next_raw;

    if (is_to_larger) {
        if (from_raw == 0x7C00) {
            return from;
        } else if (from_raw == 0xFC00) {
            return half::from_raw(0xFBFF);
        } else if (from_raw == 0xFBFF) {
            return half::from_raw(0xFC00);
        } else if (from_raw == 0x7BFF) {
            return half::from_raw(0x7C00);
        }

        next_raw = from_raw + 1;
    } else {
        if (from_raw == 0x0000) {
            return half::from_raw(0x8001);
        } else if (from_raw == 0x8000) {
            return half::from_raw(0x8001);
        } else if (from_raw == 0x7C00) {
            return half::from_raw(0x7BFF);
        } else if (from_raw == 0xFC00) {
            return from;
        }

        next_raw = from_raw - 1;
    }

    const bool sign_changed = ((from_raw ^ next_raw) & 0x8000) != 0;
    if (sign_changed) {
        next_raw = is_to_larger ? 0x7C00 : 0xFC00;
    }

    return half::from_raw(next_raw);
}

NTT_ALWAYS_INLINE half fmod(const half &a, const half &b) {
    return half::round_to_half(std::fmod(float(a), float(b)));
}
NTT_ALWAYS_INLINE half powh(const half &a, const half &b) {
    return half::round_to_half(std::pow(float(a), float(b)));
}
} // namespace nncase

namespace std {
template <> struct hash<nncase::half> {
    size_t operator()(const nncase::half &v) const {
        return hash<float>()(static_cast<float>(v));
    }
};

template <> struct numeric_limits<nncase::half> {
    static constexpr float_denorm_style has_denorm = std::denorm_present;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr bool is_bounded = false;
    static constexpr bool is_iec559 = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr float_round_style round_style = std::round_to_nearest;
    static constexpr int radix = FLT_RADIX;

    static constexpr nncase::half(min)() noexcept {
        return nncase::half::min();
    }

    static constexpr nncase::half(max)() noexcept {
        return nncase::half::highest();
    }

    static constexpr nncase::half lowest() noexcept {
        return nncase::half::lowest();
    }

    static constexpr nncase::half epsilon() noexcept {
        return nncase::half::epsilon();
    }

    static nncase::half round_error() noexcept {
        return nncase::half((double)0.5);
    }

    static constexpr nncase::half denorm_min() noexcept {
        return nncase::half::min();
    }

    static constexpr nncase::half infinity() noexcept {
        return nncase::half::infinity();
    }

    static constexpr nncase::half quiet_NaN() noexcept {
        return nncase::half::quiet_NaN();
    }

    static constexpr nncase::half signaling_NaN() noexcept {
        return nncase::half::signaling_NaN();
    }

    static constexpr int digits = 11;
    static const int min_exponent = -13;
    static const int min_exponent10 = -4;
    static const int max_exponent = 16;
    static const int max_exponent10 = 4;
};

using nncase::half;
NTT_ALWAYS_INLINE NTT_HOST_DEVICE bool isinf(const half &a) {
    return std::isinf((float)(a));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE bool isnan(const half &a) {
    return std::isnan(float(a));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE bool isfinite(const half &a) {
    return std::isfinite(float(a));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half abs(const half &a) {
    return half::round_to_half(fabsf(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half fabs(const half &a) {
    return half::round_to_half(fabs(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half exp(const half &a) {
    return half::round_to_half(expf(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half log(const half &a) {
    return half::round_to_half(logf(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half log10(const half &a) {
    return half::round_to_half(log10f(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half sqrt(const half &a) {
    return half::round_to_half(sqrtf(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half sin(const half &a) {
    return half::round_to_half(sinf(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half cos(const half &a) {
    return half::round_to_half(cosf(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half tan(const half &a) {
    return half::round_to_half(tanf(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half tanh(const half &a) {
    return half::round_to_half(tanh(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half floor(const half &a) {
    return half::round_to_half(floorf(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half ceil(const half &a) {
    return half::round_to_half(ceilf(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half round(const half &a) {
    return half::round_to_half(roundf(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half nearbyint(const half &a) {
    return half::round_to_half(nearbyintf(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half acos(const half &a) {
    return half::round_to_half(std::acos(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half asin(const half &a) {
    return half::round_to_half(std::asin(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half cosh(const half &a) {
    return half::round_to_half(std::cosh(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half sinh(const half &a) {
    return half::round_to_half(std::sinh(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE half erf(const half &a) {
    return half::round_to_half(std::erff(float(a)));
}
NTT_ALWAYS_INLINE NTT_HOST_DEVICE long lrint(const half &a) {
    return lrintf(float(a));
}

template <> struct is_floating_point<half> : public std::true_type {};
template <> struct is_arithmetic<half> : public true_type {};
} // namespace std
