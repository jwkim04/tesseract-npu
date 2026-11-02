// Test-only cycle model of the original independent per-cluster broadcast pipes.
// Uses plain history registers, not tsr_vpipe or the production head/tail split.
module tsr_mvm_bc_observer #(
  parameter int unsigned NUM_CH = 4,
  parameter int unsigned DEPTH = 2,
  parameter int unsigned RST_LOCAL = 0,
  parameter int unsigned W = 527
) (
  input logic clk,
  input logic rst_n,
  input logic top_rst_n,
  input logic [NUM_CH-1:0] cluster_rst_n,
  input logic [2:0] source_valid,
  input logic [W-1:0] source_data,
  input logic [NUM_CH-1:0][2:0] cluster_valid,
  input logic [NUM_CH-1:0][W-1:0] cluster_data,
  output logic [NUM_CH-1:0][2:0] reset_discarded
);
  logic expected_rst_n;
  if (RST_LOCAL == 0) begin : g_reset_wire
    assign expected_rst_n = rst_n;
  end else begin : g_reset_history
    logic [RST_LOCAL-1:0] history;
    initial history = '0;
    always @(posedge clk) history <= (history << 1) | RST_LOCAL'(rst_n);
    assign expected_rst_n = history[RST_LOCAL-1];
  end

  always @(posedge clk) begin
    assert (top_rst_n == expected_rst_n)
      else $fatal(1, "%m: top reset differs from independent reset history");
    assert (cluster_rst_n == {NUM_CH{expected_rst_n}})
      else $fatal(1, "%m: cluster resets differ from independent reset history");
  end

  for (genvar c = 0; c < NUM_CH; c++) begin : g_reference
    if (DEPTH == 0) begin : g_wire
      assign reset_discarded[c] = '0;
      always @(posedge clk) begin
        assert (cluster_valid[c] == source_valid)
          else $fatal(1, "%m: zero-depth valid mapping differs");
        if (|source_valid) assert (cluster_data[c] == source_data)
          else $fatal(1, "%m: zero-depth payload differs");
      end
    end else begin : g_history
      logic [DEPTH-1:0][2:0] valid_history;
      logic [DEPTH-1:0][W-1:0] data_history;
      logic armed;
      initial armed = 1'b0;
      logic [2:0] pending;
      initial reset_discarded[c] = '0;
      always_comb begin
        pending = '0;
        for (int unsigned k = 0; k < DEPTH; k++) pending |= valid_history[k];
      end
      always @(posedge clk) begin
        if (!expected_rst_n) armed <= 1'b1;
        if (!expected_rst_n && armed) reset_discarded[c] <= reset_discarded[c] | pending;
        if (!expected_rst_n) valid_history <= '0;
        else begin
          valid_history[0] <= source_valid;
          for (int unsigned k = 1; k < DEPTH; k++) valid_history[k] <= valid_history[k-1];
        end
        data_history[0] <= source_data;
        for (int unsigned k = 1; k < DEPTH; k++) data_history[k] <= data_history[k-1];
        if (armed) begin
          assert (cluster_valid[c] == valid_history[DEPTH-1])
            else $fatal(1, "%m: broadcast valid differs from original %0d-cycle pipe", DEPTH);
          if (|valid_history[DEPTH-1]) assert (cluster_data[c] == data_history[DEPTH-1])
            else $fatal(1, "%m: broadcast payload differs from original %0d-cycle pipe", DEPTH);
        end
      end
    end
  end
endmodule
