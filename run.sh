#!/bin/zsh
iverilog -o sim.out -g2012 Verilog/*.v Verilog/tb_BNN.sv &&
time vvp sim.out || exit 1
gtkwave dump.vcd
