import re
import sys
import matplotlib.pyplot as plt

# Read log from file (default: training_log.txt) or first CLI argument
log_file = sys.argv[1] if len(sys.argv) > 1 else "training_log.txt"
with open(log_file, "r") as f:
    log_text = f.read()

# Regex dopasowujący nowy format logów
pattern = re.compile(
    r"Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) \| (?:G )?Val Loss: ([\d.]+) \| (?:D Val Loss: [\d.]+ \| )?Train MSE: ([\d.]+) \| Val MSE: ([\d.]+) \| Train SSIM: ([\d.]+) \| Val SSIM: ([\d.]+) \| Train PSNR: ([\d.]+) \| Val PSNR: ([\d.]+) \| Train Corr: ([\d.]+) \| Val Corr: ([\d.]+)"
)

epochs = []
train_losses, val_losses = [], []
mse_train_losses, mse_val_losses = [], []
ssim_train_losses, ssim_val_losses = [], []
psnr_train_losses, psnr_val_losses = [], []
correlation_train_losses, correlation_val_losses = [], []

for match in pattern.finditer(log_text):
    epochs.append(int(match.group(1)))
    train_losses.append(float(match.group(2)))
    val_losses.append(float(match.group(3)))
    mse_train_losses.append(float(match.group(4)))
    mse_val_losses.append(float(match.group(5)))
    ssim_train_losses.append(float(match.group(6)))
    ssim_val_losses.append(float(match.group(7)))
    psnr_train_losses.append(float(match.group(8)))
    psnr_val_losses.append(float(match.group(9)))
    correlation_train_losses.append(float(match.group(10)))
    correlation_val_losses.append(float(match.group(11)))

if not train_losses:
    print(f"No matching log entries found in '{log_file}'.")
    sys.exit(1)

epochs = range(1, len(train_losses) + 1)

# ================= LOSS =================
plt.figure()
plt.plot(epochs, train_losses, label="Train Total Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()

# ================= MSE =================
plt.figure()
plt.plot(epochs, mse_train_losses, label="Train MSE")
plt.plot(epochs, mse_val_losses, label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE")
plt.legend()
plt.grid()
plt.show()

# ================= SSIM =================
plt.figure()
plt.plot(epochs, ssim_train_losses, label="Train SSIM")
plt.plot(epochs, ssim_val_losses, label="Val SSIM")
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.title("SSIM")
plt.legend()
plt.grid()
plt.show()

# ================= PSNR =================
plt.figure()
plt.plot(epochs, psnr_train_losses, label="Train PSNR")
plt.plot(epochs, psnr_val_losses, label="Val PSNR")
plt.xlabel("Epoch")
plt.ylabel("PSNR")
plt.title("PSNR")
plt.legend()
plt.grid()
plt.show()

# ================= CORRELATION =================
plt.figure()
plt.plot(epochs, correlation_train_losses, label="Train Corr")
plt.plot(epochs, correlation_val_losses, label="Val Corr")
plt.xlabel("Epoch")
plt.ylabel("Correlation")
plt.title("Correlation")
plt.legend()
plt.grid()
plt.show()
