# Binarized Neural Network on FPGA

FPGA-based BNN implementation project
Transplant the BNN model learned with PyTorch to Verilog to classify MNIST data

## 🎯 Purpose
To explore the feasibility of real-time handwritten digit classfication using BNNs on FPGA

## Highlight
- Training Binarization Models for PyTorch → Verilog Porting
- FC Layer-based Inference Pipeline (XNOR + Popcount + Threshold)
- BatchNorm reflected threshold correction
- Testbench simulation accuracy: **73.45%**

## Structure
1. PyTorch BNN Training
2. Weight/Threshold Export (.txt)
3. Verilog-based Inference (FC1 -> FC2 -> Argmax)
4. Simulation Testbench

## File Description
- `BNN.py` : PyTorch-based BNN training code
- `fc1_weight_bin.txt` : FC1 layer binarization weight
- `tb_BNN.v` : Verilog testbench (SystemVerilog)
- `accuracy_log.txt` : Cumulative accuracy record during testbench simulation

## Performance
- PyTorch training accuracy : **86.94%**
- Verilog simulation accuracy : **73.45%**
- dataset : MNIST (test-dataset 10000)

## Result Graph
![Verilog Accuracy Graph](Images/accuracy_graph.png)
![Comparison Graph](Images/comparison_graph.png)
![PyTorch Train Graph](Images/train_graph.png)

## License
MIT License

## Developed by.
- Jo Yonggi
- Korea Aerospace Univ. 2nd Grade
- Department of Aviation Electronics and Information Engineering
- GitHub: [@CourageHW](https://github.com/CourageHW)