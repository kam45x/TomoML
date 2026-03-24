import re
import matplotlib.pyplot as plt

log_text = """
Epoch 1/31 | Train Loss: 5.6254 | Val Loss: 4.4367 | Train MSE: 0.004212 | Val MSE: 0.002469 | Train SSIM: 0.7276 | Val SSIM: 0.7841 | Train PSNR: 26.3331 | Val PSNR: 28.2379 | Train Corr: 0.8905 | Val Corr: 0.9351
Epoch 2/31 | Train Loss: 4.0151 | Val Loss: 4.0871 | Train MSE: 0.001992 | Val MSE: 0.002059 | Train SSIM: 0.8067 | Val SSIM: 0.8170 | Train PSNR: 29.3137 | Val PSNR: 28.8298 | Train Corr: 0.9474 | Val Corr: 0.9452
Epoch 3/31 | Train Loss: 3.6688 | Val Loss: 3.6791 | Train MSE: 0.001662 | Val MSE: 0.001670 | Train SSIM: 0.8310 | Val SSIM: 0.8350 | Train PSNR: 30.2345 | Val PSNR: 30.0149 | Train Corr: 0.9562 | Val Corr: 0.9560
Epoch 4/31 | Train Loss: 3.4886 | Val Loss: 3.6582 | Train MSE: 0.001499 | Val MSE: 0.001674 | Train SSIM: 0.8441 | Val SSIM: 0.8439 | Train PSNR: 30.7627 | Val PSNR: 30.2725 | Train Corr: 0.9605 | Val Corr: 0.9578
Epoch 5/31 | Train Loss: 3.3689 | Val Loss: 3.5700 | Train MSE: 0.001394 | Val MSE: 0.001518 | Train SSIM: 0.8527 | Val SSIM: 0.8487 | Train PSNR: 31.1295 | Val PSNR: 30.2954 | Train Corr: 0.9633 | Val Corr: 0.9599
Epoch 6/31 | Train Loss: 3.2811 | Val Loss: 5.3725 | Train MSE: 0.001319 | Val MSE: 0.003839 | Train SSIM: 0.8590 | Val SSIM: 0.8137 | Train PSNR: 31.4055 | Val PSNR: 25.3390 | Train Corr: 0.9653 | Val Corr: 0.8986
Epoch 7/31 | Train Loss: 3.2125 | Val Loss: 3.6028 | Train MSE: 0.001261 | Val MSE: 0.001557 | Train SSIM: 0.8639 | Val SSIM: 0.8528 | Train PSNR: 31.6249 | Val PSNR: 30.3915 | Train Corr: 0.9668 | Val Corr: 0.9597
Epoch 8/31 | Train Loss: 3.1563 | Val Loss: 3.3517 | Train MSE: 0.001214 | Val MSE: 0.001380 | Train SSIM: 0.8679 | Val SSIM: 0.8621 | Train PSNR: 31.8087 | Val PSNR: 31.1661 | Train Corr: 0.9681 | Val Corr: 0.9640
Epoch 9/31 | Train Loss: 3.1092 | Val Loss: 3.4661 | Train MSE: 0.001175 | Val MSE: 0.001418 | Train SSIM: 0.8712 | Val SSIM: 0.8621 | Train PSNR: 31.9634 | Val PSNR: 30.6183 | Train Corr: 0.9691 | Val Corr: 0.9628
Epoch 10/31 | Train Loss: 3.0682 | Val Loss: 3.3659 | Train MSE: 0.001141 | Val MSE: 0.001381 | Train SSIM: 0.8740 | Val SSIM: 0.8652 | Train PSNR: 32.0985 | Val PSNR: 31.0699 | Train Corr: 0.9700 | Val Corr: 0.9641
Epoch 11/31 | Train Loss: 3.0324 | Val Loss: 3.7218 | Train MSE: 0.001111 | Val MSE: 0.001826 | Train SSIM: 0.8765 | Val SSIM: 0.8524 | Train PSNR: 32.2187 | Val PSNR: 30.0801 | Train Corr: 0.9708 | Val Corr: 0.9527
Epoch 12/31 | Train Loss: 2.9999 | Val Loss: 3.5236 | Train MSE: 0.001085 | Val MSE: 0.001460 | Train SSIM: 0.8787 | Val SSIM: 0.8617 | Train PSNR: 32.3317 | Val PSNR: 30.3269 | Train Corr: 0.9715 | Val Corr: 0.9614
Epoch 13/31 | Train Loss: 2.9689 | Val Loss: 3.3470 | Train MSE: 0.001060 | Val MSE: 0.001336 | Train SSIM: 0.8808 | Val SSIM: 0.8669 | Train PSNR: 32.4421 | Val PSNR: 30.8507 | Train Corr: 0.9721 | Val Corr: 0.9649
Epoch 14/31 | Train Loss: 2.9375 | Val Loss: 3.3067 | Train MSE: 0.001036 | Val MSE: 0.001321 | Train SSIM: 0.8830 | Val SSIM: 0.8711 | Train PSNR: 32.5571 | Val PSNR: 31.1439 | Train Corr: 0.9728 | Val Corr: 0.9655
Epoch 15/31 | Train Loss: 2.9045 | Val Loss: 3.1749 | Train MSE: 0.001015 | Val MSE: 0.001259 | Train SSIM: 0.8852 | Val SSIM: 0.8735 | Train PSNR: 32.6757 | Val PSNR: 31.7314 | Train Corr: 0.9733 | Val Corr: 0.9673
Epoch 16/31 | Train Loss: 2.8773 | Val Loss: 3.1143 | Train MSE: 0.000996 | Val MSE: 0.001223 | Train SSIM: 0.8868 | Val SSIM: 0.8745 | Train PSNR: 32.7648 | Val PSNR: 31.8441 | Train Corr: 0.9738 | Val Corr: 0.9679
Epoch 17/31 | Train Loss: 2.8544 | Val Loss: 3.1963 | Train MSE: 0.000978 | Val MSE: 0.001252 | Train SSIM: 0.8883 | Val SSIM: 0.8745 | Train PSNR: 32.8424 | Val PSNR: 31.3656 | Train Corr: 0.9743 | Val Corr: 0.9671
Epoch 18/31 | Train Loss: 2.8346 | Val Loss: 4.8347 | Train MSE: 0.000963 | Val MSE: 0.002851 | Train SSIM: 0.8896 | Val SSIM: 0.8378 | Train PSNR: 32.9109 | Val PSNR: 26.5980 | Train Corr: 0.9747 | Val Corr: 0.9246
Epoch 19/31 | Train Loss: 2.8144 | Val Loss: 3.4340 | Train MSE: 0.000947 | Val MSE: 0.001367 | Train SSIM: 0.8910 | Val SSIM: 0.8712 | Train PSNR: 32.9785 | Val PSNR: 30.6941 | Train Corr: 0.9751 | Val Corr: 0.9643
Epoch 20/31 | Train Loss: 2.7961 | Val Loss: 3.4372 | Train MSE: 0.000932 | Val MSE: 0.001424 | Train SSIM: 0.8921 | Val SSIM: 0.8717 | Train PSNR: 33.0412 | Val PSNR: 30.5622 | Train Corr: 0.9755 | Val Corr: 0.9625
Epoch 21/31 | Train Loss: 2.7797 | Val Loss: 3.1830 | Train MSE: 0.000920 | Val MSE: 0.001241 | Train SSIM: 0.8932 | Val SSIM: 0.8766 | Train PSNR: 33.0990 | Val PSNR: 31.4425 | Train Corr: 0.9758 | Val Corr: 0.9678
Epoch 22/31 | Train Loss: 2.7624 | Val Loss: 3.0746 | Train MSE: 0.000906 | Val MSE: 0.001209 | Train SSIM: 0.8942 | Val SSIM: 0.8778 | Train PSNR: 33.1577 | Val PSNR: 31.9312 | Train Corr: 0.9762 | Val Corr: 0.9686
"""

pattern = re.compile(
    r"Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+) \| Train MSE: ([\d.]+) \| Val MSE: ([\d.]+) \| Train SSIM: ([\d.]+) \| Val SSIM: ([\d.]+) \| Train PSNR: ([\d.]+) \| Val PSNR: ([\d.]+) \| Train Corr: ([\d.]+) \| Val Corr: ([\d.]+)"
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