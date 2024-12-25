import matplotlib.pyplot as plt

# Data for training time per epoch
epochs = list(range(1, 51))  # 50 epochs
time = [
    193.20, 176.76, 391.16, 268.39, 173.61, 173.05, 318.33, 376.62, 377.78, 334.00,
    381.51, 377.63, 375.34, 381.48, 384.21, 259.86, 179.52, 179.86, 336.05, 364.17,
    419.65, 372.67, 375.03, 263.91, 171.72, 172.73, 171.72, 172.26, 171.18, 172.56,
    342.74, 170.63, 172.55, 170.99, 172.03, 204.57, 347.51, 371.78, 369.21, 369.89,
    371.60, 368.13, 369.59, 368.90, 375.12, 371.90, 372.28, 370.63, 374.23, 374.19
]

# Plot Training Time Per Epoch
plt.figure(figsize=(10, 6))
plt.plot(epochs, time, label="Training Time", marker="o", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Time (seconds)")
plt.title("Training Time Per Epoch")
plt.grid()
plt.legend()
plt.show()
