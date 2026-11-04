// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Karl-Lin-sys
// Compile: g++ -std=c++20 -O3 -Wall -Wextra -Wpedantic -Werror cdsp_edge_infer.cpp -o cdsp_edge_infer

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <span>

namespace sunplus::cdsp {

// Hardware-exact Unbiased Round-Half-to-Even (IEEE 754 / Dyadic QAT compliant)
// Eliminates negative DC drift (-0.5 LSB) in dark WDR and thermal scenes.
// Time Complexity: O(1) | Space Complexity: O(1)
[[nodiscard]] constexpr std::int32_t dyadic_rne_shift(
    std::int64_t acc, std::uint16_t m0, std::uint8_t rshift) noexcept {
    if (rshift == 0U) {
        return static_cast<std::int32_t>(
            std::clamp<std::int64_t>(acc * static_cast<std::int64_t>(m0), -127LL, 127LL));
    }
    const std::int64_t prod = acc * static_cast<std::int64_t>(m0);
    const std::int64_t half = 1LL << (rshift - 1U);
    const std::int64_t mask = (1LL << rshift) - 1LL;
    const std::int64_t remainder = prod & mask;
    const std::int64_t is_odd = (prod >> rshift) & 1LL;

    const std::int64_t rounded = (prod >> rshift) +
        ((remainder > half || (remainder == half && is_odd != 0LL)) ? 1LL : 0LL);

    return static_cast<std::int32_t>(std::clamp<std::int64_t>(rounded, 0LL, 127LL));
}

// 1. Thermal Camera (LWIR) Two-Point NUC + 3x1 Dead-Pixel Morphological Clamp
// Protects INT8 dynamic range from microbolometer blinking/dead pixels.
// Time Complexity: O(H * W) | Auxiliary Space Complexity: O(1)
template <std::size_t Width, std::size_t Height>
void thermal_nuc_and_outlier_clamp(
    std::span<const std::uint16_t, Width * Height> raw_lwir_14b,
    std::span<const std::uint16_t, Width * Height> gain_q12,
    std::span<const std::int16_t, Width * Height> offset_s12,
    std::span<std::uint8_t, Width * Height> norm_out_u8) noexcept {

    for (std::size_t r = 0; r < Height; ++r) {
        for (std::size_t c = 0; c < Width; ++c) {
            const std::size_t idx = r * Width + c;
            const std::size_t left = r * Width + (c > 0 ? c - 1 : 0);
            const std::size_t right = r * Width + (c + 1 < Width ? c + 1 : Width - 1);

            std::array<std::uint16_t, 3> neighborhood = {
                raw_lwir_14b[left], raw_lwir_14b[idx], raw_lwir_14b[right]
            };
            if (neighborhood[0] > neighborhood[1]) std::swap(neighborhood[0], neighborhood[1]);
            if (neighborhood[1] > neighborhood[2]) std::swap(neighborhood[1], neighborhood[2]);
            if (neighborhood[0] > neighborhood[1]) std::swap(neighborhood[0], neighborhood[1]);
            const std::uint16_t despiked = neighborhood[1];

            const std::int32_t corrected =
                static_cast<std::int32_t>((static_cast<std::uint32_t>(despiked) * gain_q12[idx]) >> 12U) +
                static_cast<std::int32_t>(offset_s12[idx]);

            norm_out_u8[idx] = static_cast<std::uint8_t>(std::clamp(corrected >> 7, 0, 127));
        }
    }
}

// 2. Joint WDR Dual-Exposure Tone-Mapping + Motion-Adaptive TNR (Alignment-Free IIR)
// Time Complexity: O(H * W) | Auxiliary Space Complexity: O(1)
template <std::size_t Width, std::size_t Height>
void wdr_tnr_joint_fusion(
    std::span<const std::uint16_t, Width * Height> long_exp_12b,
    std::span<const std::uint16_t, Width * Height> short_exp_12b,
    std::span<std::uint16_t, Width * Height> prev_temporal_12b,
    std::span<std::int8_t, Width * Height> fused_int8_out,
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
        fused_int8_out[i] = static_cast<std::int8_t>(std::clamp<int>(tnr_out >> 5U, 0, 127));
    }
}

// 3. Line-Buffered 3x3 Fused RepConv Accelerator (Zero-Heap, Cache-Aligned)
// Mirrors RTL 3-row rolling line buffer with replicated border padding.
// Time Complexity: O(H * W * C * 9) | Space Complexity: O(3 * W * C)
template <std::size_t Width, std::size_t Height, std::size_t Channels>
struct alignas(64) LineBufferedRepConvEngine {
    std::array<std::int8_t, Channels * Channels * 9> weights{};
    std::array<std::int32_t, Channels> bias{};
    std::array<std::uint16_t, Channels> dyadic_m0{};
    std::array<std::uint8_t, Channels> dyadic_rshift{};
    alignas(64) std::array<std::int8_t, 3 * Width * Channels> line_buf{};

    void execute(
        std::span<const std::int8_t, Width * Height * Channels> input,
        std::span<std::int8_t, Width * Height * Channels> output) noexcept {

        for (std::size_t r = 0; r < Height; ++r) {
            for (std::size_t kr = 0; kr < 3; ++kr) {
                const std::size_t src_r = (r + kr == 0) ? 0 : std::min(Height - 1, r + kr - 1);
                for (std::size_t c = 0; c < Width * Channels; ++c) {
                    line_buf[kr * (Width * Channels) + c] = input[src_r * (Width * Channels) + c];
                }
            }

            for (std::size_t c = 0; c < Width; ++c) {
                const std::size_t c_left = (c > 0) ? (c - 1) : 0;
                const std::size_t c_right = (c + 1 < Width) ? (c + 1) : (Width - 1);
                const std::array<std::size_t, 3> cols = {c_left, c, c_right};

                for (std::size_t co = 0; co < Channels; ++co) {
                    std::int64_t acc = bias[co];
                    for (std::size_t ci = 0; ci < Channels; ++ci) {
                        const std::size_t w_base = (co * Channels + ci) * 9;
                        for (std::size_t kr = 0; kr < 3; ++kr) {
                            for (std::size_t kc = 0; kc < 3; ++kc) {
                                const std::int8_t px =
                                    line_buf[kr * (Width * Channels) + cols[kc] * Channels + ci];
                                const std::int8_t w = weights[w_base + kr * 3 + kc];
                                acc += static_cast<std::int64_t>(px) * static_cast<std::int64_t>(w);
                            }
                        }
                    }
                    output[(r * Width + c) * Channels + co] = static_cast<std::int8_t>(
                        dyadic_rne_shift(acc, dyadic_m0[co], dyadic_rshift[co]));
                }
            }
        }
    }
};

} // namespace sunplus::cdsp

int main() {
    constexpr std::size_t W = 16, H = 16, C = 4;
    alignas(64) std::array<std::uint16_t, W * H> long_exp{};
    alignas(64) std::array<std::uint16_t, W * H> short_exp{};
    alignas(64) std::array<std::uint16_t, W * H> prev_tnr{};
    alignas(64) std::array<std::int8_t, W * H> tnr_out{};

    long_exp.fill(3200U);
    short_exp.fill(210U);
    prev_tnr.fill(3180U);

    sunplus::cdsp::wdr_tnr_joint_fusion<W, H>(long_exp, short_exp, prev_tnr, tnr_out);

    sunplus::cdsp::LineBufferedRepConvEngine<W, H, C> engine{};
    engine.weights.fill(2);
    engine.bias.fill(16);
    engine.dyadic_m0.fill(16384U);
    engine.dyadic_rshift.fill(16U);

    alignas(64) std::array<std::int8_t, W * H * C> feat_in{};
    alignas(64) std::array<std::int8_t, W * H * C> feat_out{};
    feat_in.fill(32);

    engine.execute(feat_in, feat_out);
    std::cout << "[OK] C++20 Edge CDSP executed | Sample TNR=" << static_cast<int>(tnr_out[0])
              << " | Sample Fused RepConv=" << static_cast<int>(feat_out[0]) << "\n";
    return 0;
}
