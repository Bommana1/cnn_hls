open_project cnn_hls
set_top cnn_inference
add_files cnn_hls.cpp
add_files -tb testbench.cpp
open_solution "solution1"
set_part {xcvu108-fsvh2892-2-e}
create_clock -period 10 -name default
csynth_design
export_design -format ip_catalog
exit
