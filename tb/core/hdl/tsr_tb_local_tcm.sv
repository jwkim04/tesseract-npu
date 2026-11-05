// Test-only ELF preload. Normal core benches exercise the CSR loader.
module tsr_tb_local_tcm;
  string itcm, dtcm;
  initial begin
    if ($value$plusargs("tb_itcm=%s", itcm))
      $readmemh(itcm, $root.tsr_core.u_ctrl.u_rv.u_itcm.mem);
    if ($value$plusargs("tb_dtcm=%s", dtcm))
      $readmemh(dtcm, $root.tsr_core.u_ctrl.u_rv.u_dtcm.mem);
  end
endmodule

bind tsr_core tsr_tb_local_tcm u_tb_local_tcm();
