open_hw_manager
connect_hw_server -allow_non_jtag
puts [get_hw_targets]
disconnect_hw_server
close_hw_manager
quit

