// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Karl-Lin-sys
// Compile (C++14/17/20 compatible):
//   g++ -std=c++14 -O3 -Wall -Wextra -Wpedantic -Werror firmware/cdsp_edge_infer.cpp -o cdsp_edge_infer

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "generated_rom_weights.h"

namespace sunplus {
namespace cdsp {

// Hardware-exact Unbiased Round-Half-to-Even (IEEE 754 / Dyadic QAT compliant)
// Eliminates negative DC drift (-0.5 LSB) in dark WDR and thermal scenes.
// Time Complexity: O(1) | Space Complexity: O(1)
inline std::int32_t dyadic_rne_shift(
    std::int64_t acc, std::uint16_t m0, std::uint8_t rshift) noexcept {
    const std::int64_t prod = acc * static_cast<std::int64_t>(m0);
    std::int64_t rounded = prod;
    if (rshift > 0U) {
        const std::int64_t half = 1LL << (rshift - 1U);
        const std::int64_t mask = (1LL << rshift) - 1LL;
        const std::int64_t remainder = prod & mask;
        const std::int64_t is_odd = (prod >> rshift) & 1LL;
        rounded = (prod >> rshift) +
                  ((remainder > half || (remainder == half && is_odd != 0LL)) ? 1LL : 0LL);
    }
    if (rounded < 0LL) {
        return 0;
    }
    if (rounded > 127LL) {
        return 127;
    }
    return static_cast<std::int32_t>(rounded);
}

// 1. Thermal Camera (LWIR) Two-Point NUC + 3x1 Dead-Pixel Morphological Clamp
// Time Complexity: O(H * W) | Auxiliary Space Complexity: O(1)
template <std::size_t Width, std::size_t Height>
void thermal_nuc_and_outlier_clamp(
    const std::array<std::uint16_t, Width * Height>& raw_lwir_14b,
    const std::array<std::uint16_t, Width * Height>& gain_q12,
    const std::array<std::int16_t, Width * Height>& offset_s12,
    std::array<std::uint8_t, Width * Height>& norm_out_u8) noexcept {

    for (std::size_t r = 0; r < Height; ++r) {
        for (std::size_t c = 0; c < Width; ++c) {
            const std::size_t idx = r * Width + c;
            const std::size_t left = r * Width + (c > 0 ? c - 1 : 0);
            const std::size_t right = r * Width + (c + 1 < Width ? c + 1 : Width - 1);

            std::array<std::uint16_t, 3> neighborhood = {{
                raw_lwir_14b[left], raw_lwir_14b[idx], raw_lwir_14b[right]
            }};
            if (neighborhood[0] > neighborhood[1]) std::swap(neighborhood[0], neighborhood[1]);
            if (neighborhood[1] > neighborhood[2]) std::swap(neighborhood[1], neighborhood[2]);
            if (neighborhood[0] > neighborhood[1]) std::swap(neighborhood[0], neighborhood[1]);
            const std::uint16_t despiked = neighborhood[1];

            const std::int32_t corrected =
                static_cast<std::int32_t>((static_cast<std::uint32_t>(despiked) * gain_q12[idx]) >> 12U) +
                static_cast<std::int32_t>(offset_s12[idx]);

            const std::int32_t shifted = corrected >> 7;
            norm_out_u8[idx] = static_cast<std::uint8_t>(
                shifted < 0 ? 0 : (shifted > 127 ? 127 : shifted));
        }
    }
}

// 2. Joint WDR Dual-Exposure Tone-Mapping + Motion-Adaptive TNR (Alignment-Free IIR)
// Time Complexity: O(H * W) | Auxiliary Space Complexity: O(1)
template <std::size_t Width, std::size_t Height>
void wdr_tnr_joint_fusion(
    const std::array<std::uint16_t, Width * Height>& long_exp_12b,
    const std::array<std::uint16_t, Width * Height>& short_exp_12b,
    std::array<std::uint16_t, Width * Height>& prev_temporal_12b,
    std::array<std::int8_t, Width * Height>& fused_int8_out,
    std::uint16_t exp_ratio_q4 = 16U,
    std::uint16_t motion_thresh = 64U) noexcept {

    for (std::size_t i = 0; i < Width * Height; ++i) {
        const std::uint32_t l_px = long_exp_12b[i];
        const std::uint32_t s_px = static_cast<std::uint32_t>(short_exp_12b[i]) * exp_ratio_q4;

        std::uint32_t w_short = 0U;
        if (l_px >= 3824U) {
            w_short = 256U;
        } else if (l_px > 2800U) {
            w_short = ((l_px - 2800U) * 256U) >> 10U;
        }
        const std::uint32_t wdr_linear = ((256U - w_short) * l_px + w_short * s_px) >> 8U;

        const std::uint16_t wdr_12b = static_cast<std::uint16_t>(
            std::min<std::uint32_t>(4095U, wdr_linear >> 2U));

        const std::uint16_t hist_px = prev_temporal_12b[i];
        const std::uint16_t diff = (wdr_12b > hist_px) ? (wdr_12b - hist_px) : (hist_px - wdr_12b);
        const std::uint32_t alpha_q8 = (diff >= motion_thresh)
            ? 256U
            : ((static_cast<std::uint32_t>(diff) * 224U) / motion_thresh + 32U);

        const std::uint16_t tnr_out = static_cast<std::uint16_t>(
            (alpha_q8 * wdr_12b + (256U - alpha_q8) * hist_px) >> 8U);

        prev_temporal_12b[i] = tnr_out;
        const int scaled = static_cast<int>(tnr_out >> 5U);
        fused_int8_out[i] = static_cast<std::int8_t>(
            scaled < 0 ? 0 : (scaled > 127 ? 127 : scaled));
    }
}

// 3. Single-Window 3x3 Fused RepConv PE Golden Reference (Bit-Accurate to RTL `ai_cdsp_pe_array.sv`)
inline std::int8_t pe_array_single_window_golden(
    const std::array<std::int8_t, 9>& window,
    const std::array<std::int8_t, 9>& weights,
    std::int32_t bias,
    std::uint16_t dyadic_m0,
    std::uint8_t dyadic_rshift) noexcept {

    std::int64_t acc = static_cast<std::int64_t>(bias);
    for (std::size_t k = 0; k < 9; ++k) {
        acc += static_cast<std::int64_t>(window[k]) * static_cast<std::int64_t>(weights[k]);
    }
    return static_cast<std::int8_t>(dyadic_rne_shift(acc, dyadic_m0, dyadic_rshift));
}

} // namespace cdsp
} // namespace sunplus

int main() {
    using namespace sunplus::cdsp;
    constexpr std::size_t W = 16, H = 16;
    alignas(64) std::array<std::uint16_t, W * H> long_exp{};
    alignas(64) std::array<std::uint16_t, W * H> short_exp{};
    alignas(64) std::array<std::uint16_t, W * H> prev_tnr{};
    alignas(64) std::array<std::int8_t, W * H> tnr_out{};

    long_exp.fill(3200U);
    short_exp.fill(210U);
    prev_tnr.fill(3180U);

    wdr_tnr_joint_fusion<W, H>(long_exp, short_exp, prev_tnr, tnr_out);

    // Evaluate all compiled test windows using the weights exported by `compiler/ai_cdsp_compiler.py`
    std::ofstream out_file("cpp_golden_outputs.txt");
    std::cout << "[OK] C++ Edge CDSP Golden Model Executed (Loaded Compiler ROM SHA-256 Prefix: "
              << rom::kCompiledSha256Prefix << ")\n";

    for (std::size_t w_idx = 0; w_idx < rom::kNumTestVectors; ++w_idx) {
        const std::int8_t y = pe_array_single_window_golden(
            rom::kTestWindows[w_idx],
            rom::kCompiledKernel3x3,
            rom::kCompiledBias,
            rom::kCompiledDyadicM0,
            rom::kCompiledDyadicRshift);
        out_file << static_cast<int>(y) << "\n";
        std::cout << "     Vector[" << w_idx << "] -> C++ Golden Output = " << static_cast<int>(y) << "\n";
    }
    return 0;
}
