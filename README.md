# BNN_ver1.0

> MNIST dataset | FUlly-parallel inference | Threshold tuning | Verilog-compatible pipeline

## 🎯 Final Results
- PyTorch Float Accuracy: ** ~87.0%**
- Verilog-style Inference Accuracy: **~73.4%**
- Bit-level hardware threshold tuning using median/statistics

## 📁 Project Structure
- `verilog/`     : RTL module (`fc1.v`, `argmax.v`, ...)
- `python/`      : BNN training & threshold extraction scripts
- `verilog/data` : Binarized weights & input
- `log/`         : Accuracy logs
- `img/`         : Accuracy graph comparing PyTorch & Verilog

## 🛠️ Tools Used
- Vivado / GTKWave
- PyTorch
- Verilog / SystemVerilog

## 📌 How to Run


### 🧠 Train Model
```bash
python BNN.py