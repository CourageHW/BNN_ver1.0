import matplotlib.pyplot as plt

models = ['Float Model', 'BNN (PyTorch)', 'BNN (Verilog)']
accuracies = [86.94, 73.47, 73.45]

plt.bar(models, accuracies, color=['skyblue', 'orange', 'green'])
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison of Float / BNN / Verilog')
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 1, f'{acc:.2f}%', ha='center')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()