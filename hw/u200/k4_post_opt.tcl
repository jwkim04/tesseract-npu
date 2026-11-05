# Save the actual optimized design before the K4 pre-place floorplan gate.
# Vitis runs this user hook from the native implementation run directory,
# after its platform/native OPT.POST hooks. A failed write fails the run.
write_checkpoint -force [file join [pwd] k4_post_opt.dcp]
puts "K4 POST-OPT CHECKPOINT COMPLETE: [file join [pwd] k4_post_opt.dcp]"
