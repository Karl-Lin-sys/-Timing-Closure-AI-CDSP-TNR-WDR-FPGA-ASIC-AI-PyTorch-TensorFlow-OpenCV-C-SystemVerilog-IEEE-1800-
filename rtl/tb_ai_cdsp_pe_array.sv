// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Karl-Lin-sys
// Standard: IEEE Std 1800-2023 SystemVerilog
// Module: tb_ai_cdsp_pe_array
// Description: Self-checking Co-Simulation Testbench loading actual compiled PyTorch ROM
//              parameters (`generated_rom_vectors.svh`), dumping VCD waveforms (`ai_cdsp_wave.vcd`),
//              and writing `rtl_sim_outputs.txt` for bit-exact verification against C++ Golden Model.

`default_nettype none
`timescale 1ns / 1ps

module tb_ai_cdsp_pe_array;

    localparam int unsigned DATA_W         = 8;
    localparam int unsigned ACC_W          = 24;
    localparam int unsigned M0_W           = 15;
    localparam int unsigned SHIFT_W        = 5;
    localparam int unsigned WATCHDOG_LIMIT = 16;

    `include "rtl/generated_rom_vectors.svh"

    logic                     clk;
    logic                     rst_n;
    logic                     cfg_we;
    logic        [9*DATA_W-1:0] cfg_weights;
    logic signed [ACC_W-1:0]  cfg_bias;
    logic        [M0_W-1:0]   cfg_dyadic_m0;
    logic        [SHIFT_W-1:0] cfg_dyadic_shift;
    logic        [DATA_W-1:0] cfg_expected_parity;

    logic                     s_axis_tvalid;
    logic                     s_axis_tready;
    logic        [9*DATA_W-1:0] s_axis_window;
    logic                     s_axis_tlast;

    logic                     m_axis_tvalid;
    logic                     m_axis_tready;
    logic signed [DATA_W-1:0] m_axis_tdata;
    logic                     m_axis_tlast;

    logic                     sec_trojan_parity_err;
    logic                     sec_axi_deadlock_irq;

    ai_cdsp_pe_array #(
        .DATA_W        (DATA_W),
        .ACC_W         (ACC_W),
        .M0_W          (M0_W),
        .SHIFT_W       (SHIFT_W),
        .WATCHDOG_LIMIT(WATCHDOG_LIMIT)
    ) dut (
        .clk_i                  (clk),
        .rst_ni                 (rst_n),
        .cfg_we_i               (cfg_we),
        .cfg_weights_i          (cfg_weights),
        .cfg_bias_i             (cfg_bias),
        .cfg_dyadic_m0_i        (cfg_dyadic_m0),
        .cfg_dyadic_shift_i     (cfg_dyadic_shift),
        .cfg_expected_parity_i  (cfg_expected_parity),
        .s_axis_tvalid_i        (s_axis_tvalid),
        .s_axis_tready_o        (s_axis_tready),
        .s_axis_window_i        (s_axis_window),
        .s_axis_tlast_i         (s_axis_tlast),
        .m_axis_tvalid_o        (m_axis_tvalid),
        .m_axis_tready_i        (m_axis_tready),
        .m_axis_tdata_o         (m_axis_tdata),
        .m_axis_tlast_o         (m_axis_tlast),
        .sec_trojan_parity_err_o(sec_trojan_parity_err),
        .sec_axi_deadlock_irq_o (sec_axi_deadlock_irq)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    int fd;

    task automatic run_single_window(
        input int unsigned       vec_idx,
        input logic [71:0]       win_flat,
        input logic signed [7:0] exp_out
    );
        begin
            s_axis_tvalid = 1'b1;
            s_axis_window = win_flat;
            s_axis_tlast  = (vec_idx == 3);
            @(posedge clk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast  = 1'b0;

            wait (m_axis_tvalid === 1'b1);
            @(posedge clk);
            $fdisplay(fd, "%0d", m_axis_tdata);
            $display("     Vector[%0d] -> SystemVerilog RTL Output = %0d (Expected = %0d)",
                     vec_idx, m_axis_tdata, exp_out);
            if (m_axis_tdata !== exp_out) begin
                $display("[FAIL] Bit-accurate mismatch at Vector[%0d]! Expected %0d, got %0d",
                         vec_idx, exp_out, m_axis_tdata);
                $fatal(1);
            end
        end
    endtask

    initial begin
        $dumpfile("ai_cdsp_wave.vcd");
        $dumpvars(0, tb_ai_cdsp_pe_array);

        fd = $fopen("rtl_sim_outputs.txt", "w");

        rst_n               = 1'b0;
        cfg_we              = 1'b0;
        cfg_weights         = '0;
        cfg_bias            = '0;
        cfg_dyadic_m0       = 15'd16384;
        cfg_dyadic_shift    = 5'd14;
        cfg_expected_parity = 8'd0;
        s_axis_tvalid       = 1'b0;
        s_axis_window       = '0;
        s_axis_tlast        = 1'b0;
        m_axis_tready       = 1'b1;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        // 1. Load Compiled PyTorch Weights & Dyadic Parameters from `generated_rom_vectors.svh`
        cfg_we              = 1'b1;
        cfg_weights         = COMPILED_WEIGHTS_FLAT;
        cfg_bias            = COMPILED_BIAS;
        cfg_dyadic_m0       = COMPILED_DYADIC_M0;
        cfg_dyadic_shift    = COMPILED_DYADIC_RSHIFT;
        cfg_expected_parity = COMPILED_EXPECTED_PARITY;
        @(posedge clk);
        cfg_we = 1'b0;
        @(posedge clk);

        if (sec_trojan_parity_err !== 1'b0) begin
            $display("[FAIL] Unexpected Trojan parity error on valid compiled weights!");
            $fatal(1);
        end
        $display("[OK] Loaded PyTorch Compiled ROM into RTL Registers (Parity = 0x%02h)", COMPILED_EXPECTED_PARITY);

        // 2. Execute all 4 Co-Simulation Windows through the 4-stage pipeline
        run_single_window(0, TEST_WINDOW_0, EXPECTED_OUT_0);
        run_single_window(1, TEST_WINDOW_1, EXPECTED_OUT_1);
        run_single_window(2, TEST_WINDOW_2, EXPECTED_OUT_2);
        run_single_window(3, TEST_WINDOW_3, EXPECTED_OUT_3);

        $fclose(fd);
        $display("[PASS] All %0d Co-Simulation Windows Bit-Exact Verified!", NUM_CO_SIM_VECTORS);

        // 3. Verify NIST CSF 2.0 Hardware Trojan Parity Mismatch Detection
        cfg_we              = 1'b1;
        cfg_expected_parity = COMPILED_EXPECTED_PARITY ^ 8'hFF; // Inject corrupted parity
        @(posedge clk);
        cfg_we              = 1'b0;
        @(posedge clk);
        if (sec_trojan_parity_err !== 1'b1) begin
            $display("[FAIL] Trojan parity detector failed to flag mismatch!");
            $fatal(1);
        end
        $display("[PASS] NIST CSF 2.0 Hardware Trojan Weight Parity Detector Verified (sec_trojan_parity_err = 1)");

        // 4. Verify AXI4-Stream Deadlock Watchdog Timer (`sec_axi_deadlock_irq_o`)
        s_axis_tvalid = 1'b1;
        s_axis_window = TEST_WINDOW_0;
        @(posedge clk);
        s_axis_tvalid = 1'b0;
        wait (m_axis_tvalid === 1'b1);
        m_axis_tready = 1'b0; // Stall downstream sink intentionally
        repeat (WATCHDOG_LIMIT + 2) @(posedge clk);
        if (sec_axi_deadlock_irq !== 1'b1) begin
            $display("[FAIL] AXI Deadlock Watchdog failed to trigger!");
            $fatal(1);
        end
        $display("[PASS] AXI4-Stream Deadlock Watchdog Verified (sec_axi_deadlock_irq = 1 after %0d stall cycles)", WATCHDOG_LIMIT);
        m_axis_tready = 1'b1;
        @(posedge clk);

        $display("[OK] Waveform saved to ai_cdsp_wave.vcd | All IEEE 1800-2023 Assertions Passed.");
        $finish;
    end

endmodule

`default_nettype wire
