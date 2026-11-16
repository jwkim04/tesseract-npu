# hw/u200/dma_clk_margin.tcl — a setup margin on the shell's host data clock during placement and routing.
#
# The host data path of the platform runs on dma_ip_axi_aclk (4.0 ns, the XDMA's AXI clock). Its paths from the
# reconfigurable part into the isolation pipe of the static region carry about 2.6 ns of locked routing, so their driver
# cells must sit near their partition pins, and the placer's estimate leaves them no margin: this clock is the worst
# one of every K4 link, a few tens of picoseconds on either side of zero, while the kernel clocks meet.
# tsr_dma_margin::on asks for TSR_DMA_MARGIN_NS (default 0.060) more setup slack on that clock. impl_pre.tcl calls it
# before opt_design, so opt_design, place_design, the post-place phys_opt_design and route_design work against the
# tighter requirement. tsr_dma_margin::off removes it again; cdc_check.tcl calls it after route_design, so the post-route
# optimization, the frequency scaling and the final timing check of the flow see the platform's own constraint.
# It is a timing requirement, not a placement constraint: no cell, site or Pblock is named.
# TSR_DMA_MARGIN_NS=0 in the environment switches it off.
namespace eval tsr_dma_margin {
  proc ns {} {
    if {[info exists ::env(TSR_DMA_MARGIN_NS)]} { return $::env(TSR_DMA_MARGIN_NS) }
    return 0.060
  }
  proc clocks {} {
    return [get_clocks -quiet -filter {NAME =~ dma_ip_axi_aclk*}]
  }
  proc worst {} {
    set c [clocks]
    if {[llength $c] == 0} { return "-" }
    set p [get_timing_paths -quiet -setup -max_paths 1 -nworst 1 -to $c]
    if {[llength $p] == 0} { return "-" }
    return [get_property SLACK $p]
  }
  proc on {} {
    set m [ns]
    set c [clocks]
    if {$m <= 0 || [llength $c] == 0} {
      puts "DMAMARGIN off (margin $m ns, clocks: $c)"
      return
    }
    set_clock_uncertainty -setup $m $c
    puts "DMAMARGIN on: $m ns more setup slack asked on $c"
  }
  proc off {} {
    set c [clocks]
    if {[ns] <= 0 || [llength $c] == 0} { return }
    set before "-"
    set after "-"
    catch {set before [worst]}
    set_clock_uncertainty -setup 0 $c
    catch {set after [worst]}
    puts "DMAMARGIN off after route_design: worst setup slack to $c with the margin $before ns, without it $after ns"
  }
}
