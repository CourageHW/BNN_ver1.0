import matplotlib.pyplot as plt
import numpy as np

# txt 파일 읽기
progress = []
accuracy = []

with open("accuracy_log.txt", "r") as f:
    for line in f:
        p, acc = line.strip().split()
        progress.append(int(p))
        accuracy.append(float(acc))

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(progress, accuracy, label="Verilog Accuracy (%)")
plt.xlabel("Image Index")
plt.ylabel("Accuracy (%)")
plt.ylim(60, 85)
plt.axhline(y=73.45, color='gray', linestyle='--', linewidth=1)
plt.annotate('Final Accuracy: 73.45%', 
             xy=(1600, 73.45), 
             xytext=(600, 77),
             fontsize=10, color='gray',
             arrowprops=dict(arrowstyle='->', color='gray'))
plt.title("Verilog Inference Accuracy Over Time", fontsize=14, weight='bold')
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("verilog_accuracy_curve.png")  # PNG로 저장도 가능
plt.show()
