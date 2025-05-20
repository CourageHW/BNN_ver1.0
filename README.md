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

## 구조
1. PyTorch BNN Training
2. Weight/Threshold Export (.txt)
3. Verilog-based Inference (FC1 -> FC2 -> Argmax)
4. Simulation Testbench

## 주요 파일
- `BNN.py` : PyTorch-based BNN training code
- `fc1_weight_bin.txt` : FC1 layer binarization weight
- `tb_BNN.v` : Verilog testbench (SystemVerilog)
- `accuracy_log.txt` : Cumulative accuracy record during testbench simulation

## 성능
- PyTorch training accuracy : **86.94%**
- Verilog simulation accuracy : **73.45%**
- dataset : MNIST (test-dataset 10000)

## 결과 예시
![Verilog Accuracy Graph](images/accuracy_graph.png)
![Comparison Graph](images/comparison_graph.png)
![PyTorch Train Graph](train_graph.png)

## License
MIT License

## Developed by.
- Jo Yonggi
- Korea Aerospace Univ. 2nd Grade
- Department of Aviation Electronics and Information Engineering
- GitHub: [@CourageHW](https://github.com/CourageHW)