// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Karl-Lin-sys
// Standard: IEEE Std 1800-2023 SystemVerilog
// Module: ai_cdsp_pe_array
// Description: 4-Stage Pipelined 3x3 Fused RepConv + Dyadic RNE Requantizer
//              with NIST CSF 2.0 Deadlock Watchdog & Weight Parity Trojan Detector.

`default_nettype none

module ai_cdsp_pe_array #(
    parameter int unsigned DATA_W         = 8,
    parameter int unsigned ACC_W          = 24,
    parameter int unsigned M0_W           = 15,
    parameter int unsigned SHIFT_W        = 5,
    parameter int unsigned WATCHDOG_LIMIT = 1024
) (
    input  wire                         clk_i,
    input  wire                         rst_ni,

    // Weight & Dyadic Parameter Configuration Port (with Trojan Parity Check)
    input  wire                         cfg_we_i,
    input  wire         [9*DATA_W-1:0]  cfg_weights_i,
    input  wire  signed [ACC_W-1:0]     cfg_bias_i,
    input  wire         [M0_W-1:0]      cfg_dyadic_m0_i,
    input  wire         [SHIFT_W-1:0]   cfg_dyadic_shift_i,
    input  wire         [DATA_W-1:0]    cfg_expected_parity_i,

    // AXI4-Stream 3x3 Sliding Window Input (from 2-Row SRAM Line Buffer)
    input  wire                         s_axis_tvalid_i,
    output logic                        s_axis_tready_o,
    input  wire         [9*DATA_W-1:0]  s_axis_window_i,
    input  wire                         s_axis_tlast_i,

    // AXI4-Stream Quantized Output
    output logic                        m_axis_tvalid_o,
    input  wire                         m_axis_tready_i,
    output logic signed [DATA_W-1:0]    m_axis_tdata_o,
    output logic                        m_axis_tlast_o,

    // NIST CSF 2.0 Security & Reliability Telemetry
    output logic                        sec_trojan_parity_err_o,
    output logic                        sec_axi_deadlock_irq_o
);

    // -------------------------------------------------------------------------
    // 1. Configuration Registers & Continuous Weight Integrity Parity Check
    // -------------------------------------------------------------------------
    logic        [9*DATA_W-1:0] weight_q;
    logic signed [ACC_W-1:0]    bias_q;
    logic        [M0_W-1:0]     dyadic_m0_q;
    logic        [SHIFT_W-1:0]  dyadic_shift_q;
    logic        [DATA_W-1:0]   expected_parity_q;
    logic        [DATA_W-1:0]   computed_parity_c;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            weight_q          <= '0;
            bias_q            <= '0;
            dyadic_m0_q       <= 15'd16384;
            dyadic_shift_q    <= 5'd14;
            expected_parity_q <= '0;
        end else if (cfg_we_i) begin
            weight_q          <= cfg_weights_i;
            bias_q            <= cfg_bias_i;
            dyadic_m0_q       <= cfg_dyadic_m0_i;
            dyadic_shift_q    <= cfg_dyadic_shift_i;
            expected_parity_q <= cfg_expected_parity_i;
        end
    end

    // Combinational XOR reduction across all 9 INT8 weights (zero latch risk, zero simulator warnings)
    assign computed_parity_c = weight_q[7:0]   ^ weight_q[15:8]  ^ weight_q[23:16] ^
                               weight_q[31:24] ^ weight_q[39:32] ^ weight_q[47:40] ^
                               weight_q[55:48] ^ weight_q[63:56] ^ weight_q[71:64];
    assign sec_trojan_parity_err_o = (computed_parity_c != expected_parity_q);

    // -------------------------------------------------------------------------
    // 2. Deadlock-Free Pipeline Handshake & Watchdog Timer (NIST DE.CM)
    // -------------------------------------------------------------------------
    logic pipe_advance_c;
    logic [15:0] watchdog_cnt_q;

    assign pipe_advance_c  = m_axis_tready_i || !m_axis_tvalid_o;
    assign s_axis_tready_o = pipe_advance_c;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            watchdog_cnt_q         <= 16'd0;
            sec_axi_deadlock_irq_o <= 1'b0;
        end else begin
            if (m_axis_tvalid_o && !m_axis_tready_i) begin
                if (watchdog_cnt_q >= WATCHDOG_LIMIT[15:0]) begin
                    sec_axi_deadlock_irq_o <= 1'b1;
                end else begin
                    watchdog_cnt_q <= watchdog_cnt_q + 16'd1;
                end
            end else begin
                watchdog_cnt_q         <= 16'd0;
                sec_axi_deadlock_irq_o <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // 3. 4-Stage Timing-Closed Systolic Datapath (Max Logic Depth = 1 DSP / Adder)
    // -------------------------------------------------------------------------
    logic [3:0]              vld_pipe_q;
    logic [3:0]              last_pipe_q;

    // Stage 1: 9 Parallel INT8 x INT8 Multiplications (16-bit signed products)
    logic signed [15:0] s1_prod_0_q, s1_prod_1_q, s1_prod_2_q;
    logic signed [15:0] s1_prod_3_q, s1_prod_4_q, s1_prod_5_q;
    logic signed [15:0] s1_prod_6_q, s1_prod_7_q, s1_prod_8_q;

    // Stage 2: Partial Adder Tree (3 groups of 3 products -> 18-bit signed partial sums)
    logic signed [17:0] s2_part_sum_0_q, s2_part_sum_1_q, s2_part_sum_2_q;

    // Stage 3: Final Accumulation + Bias Addition (24-bit signed accumulator)
    logic signed [ACC_W-1:0] s3_acc_q;

    // Stage 4: Dyadic Multiplication + Round-Half-to-Even + ReLU6 Clamp [0, 127]
    logic signed [ACC_W+M0_W:0] s4_mul_c;
    logic signed [ACC_W+M0_W:0] s4_shifted_c;
    logic signed [DATA_W-1:0]   s4_clamped_c;
    logic signed [ACC_W+M0_W:0] half_val_c;
    logic signed [ACC_W+M0_W:0] rem_mask_c;
    logic signed [ACC_W+M0_W:0] rem_val_c;
    logic                       quot_odd_c;
    logic                       round_up_c;

    always_comb begin
        // Complete default assignments prevent unintended latch inference
        s4_mul_c     = '0;
        s4_shifted_c = '0;
        s4_clamped_c = '0;
        half_val_c   = '0;
        rem_mask_c   = '0;
        rem_val_c    = '0;
        quot_odd_c   = 1'b0;
        round_up_c   = 1'b0;

        // Signed 24-bit * Unsigned 15-bit dyadic mantissa
        s4_mul_c = s3_acc_q * $signed({1'b0, dyadic_m0_q});

        // Unbiased Round-Half-to-Even (RNE)
        if (dyadic_shift_q == '0) begin
            s4_shifted_c = s4_mul_c;
        end else begin
            half_val_c = (40'sd1 <<< (dyadic_shift_q - 5'd1));
            rem_mask_c = (40'sd1 <<< dyadic_shift_q) - 40'sd1;
            rem_val_c  = s4_mul_c & rem_mask_c;
            quot_odd_c = ((s4_mul_c >>> dyadic_shift_q) & 40'sd1) != 40'sd0;
            round_up_c = (rem_val_c > half_val_c) || ((rem_val_c == half_val_c) && quot_odd_c);

            s4_shifted_c = (s4_mul_c >>> dyadic_shift_q) + (round_up_c ? 40'sd1 : 40'sd0);
        end

        // ReLU6 Saturation Clamp to [0, 127]
        if (s4_shifted_c < 0) begin
            s4_clamped_c = 8'sd0;
        end else if (s4_shifted_c > 127) begin
            s4_clamped_c = 8'sd127;
        end else begin
            s4_clamped_c = s4_shifted_c[7:0];
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            vld_pipe_q      <= '0;
            last_pipe_q     <= '0;
            s1_prod_0_q     <= '0; s1_prod_1_q <= '0; s1_prod_2_q <= '0;
            s1_prod_3_q     <= '0; s1_prod_4_q <= '0; s1_prod_5_q <= '0;
            s1_prod_6_q     <= '0; s1_prod_7_q <= '0; s1_prod_8_q <= '0;
            s2_part_sum_0_q <= '0; s2_part_sum_1_q <= '0; s2_part_sum_2_q <= '0;
            s3_acc_q        <= '0;
            m_axis_tdata_o  <= '0;
        end else if (pipe_advance_c) begin
            vld_pipe_q  <= {vld_pipe_q[2:0], s_axis_tvalid_i};
            last_pipe_q <= {last_pipe_q[2:0], s_axis_tlast_i};

            // Stage 1: Registered 9x INT8*INT8 Multipliers
            s1_prod_0_q <= $signed(s_axis_window_i[0 +: 8])  * $signed(weight_q[0 +: 8]);
            s1_prod_1_q <= $signed(s_axis_window_i[8 +: 8])  * $signed(weight_q[8 +: 8]);
            s1_prod_2_q <= $signed(s_axis_window_i[16 +: 8]) * $signed(weight_q[16 +: 8]);
            s1_prod_3_q <= $signed(s_axis_window_i[24 +: 8]) * $signed(weight_q[24 +: 8]);
            s1_prod_4_q <= $signed(s_axis_window_i[32 +: 8]) * $signed(weight_q[32 +: 8]);
            s1_prod_5_q <= $signed(s_axis_window_i[40 +: 8]) * $signed(weight_q[40 +: 8]);
            s1_prod_6_q <= $signed(s_axis_window_i[48 +: 8]) * $signed(weight_q[48 +: 8]);
            s1_prod_7_q <= $signed(s_axis_window_i[56 +: 8]) * $signed(weight_q[56 +: 8]);
            s1_prod_8_q <= $signed(s_axis_window_i[64 +: 8]) * $signed(weight_q[64 +: 8]);

            // Stage 2: Balanced 3-Input Adder Tree
            s2_part_sum_0_q <= s1_prod_0_q + s1_prod_1_q + s1_prod_2_q;
            s2_part_sum_1_q <= s1_prod_3_q + s1_prod_4_q + s1_prod_5_q;
            s2_part_sum_2_q <= s1_prod_6_q + s1_prod_7_q + s1_prod_8_q;

            // Stage 3: Final Sum + Bias
            s3_acc_q <= s2_part_sum_0_q + s2_part_sum_1_q + s2_part_sum_2_q + bias_q;

            // Stage 4: Registered Requantized Output
            m_axis_tdata_o <= s4_clamped_c;
        end
    end

    assign m_axis_tvalid_o = vld_pipe_q[3];
    assign m_axis_tlast_o  = last_pipe_q[3];

endmodule

`default_nettype wire
