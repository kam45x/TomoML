import re
import matplotlib.pyplot as plt

DISPLAY_POINTS = 31

LOG_FILES = [
    ("logs/training_log1804.txt", "ConvNeXt"),
]

pattern = re.compile(
    r"Epoch (\d+)/\d+ \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+) \| Train MSE: ([\d.]+) \| Val MSE: ([\d.]+) \| Train SSIM: ([\d.]+) \| Val SSIM: ([\d.]+) \| Train PSNR: ([\d.]+) \| Val PSNR: ([\d.]+) \| Train Corr: ([\d.]+) \| Val Corr: ([\d.]+)"
)

logs = {}
for log_file, name in LOG_FILES:
    with open(log_file, "r") as f:
        log_text = f.read()

    data = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": [],
        "train_ssim": [],
        "val_ssim": [],
        "train_psnr": [],
        "val_psnr": [],
        "train_corr": [],
        "val_corr": [],
    }

    for match in pattern.finditer(log_text):
        data["train_loss"].append(float(match.group(2)))
        data["val_loss"].append(float(match.group(3)))
        data["train_mse"].append(float(match.group(4)))
        data["val_mse"].append(float(match.group(5)))
        data["train_ssim"].append(float(match.group(6)))
        data["val_ssim"].append(float(match.group(7)))
        data["train_psnr"].append(float(match.group(8)))
        data["val_psnr"].append(float(match.group(9)))
        data["train_corr"].append(float(match.group(10)))
        data["val_corr"].append(float(match.group(11)))

    data["epochs"] = range(1, len(data["train_loss"]) + 1)
    logs[name] = data

metrics = [
    ("mse", "MSE"),
    ("ssim", "SSIM"),
    ("psnr", "PSNR"),
    ("corr", "Correlation"),
]

for metric_key, metric_label in metrics:
    train_key = f"train_{metric_key}"
    val_key = f"val_{metric_key}"

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 7))

    for name, data in logs.items():
        label = name
        epochs = list(data["epochs"])[:DISPLAY_POINTS]
        ax_train.plot(
            epochs,
            data[train_key][:DISPLAY_POINTS],
            label=label,
            linewidth=2,
        )
        ax_val.plot(
            epochs,
            data[val_key][:DISPLAY_POINTS],
            label=label,
            linewidth=2,
        )

    ax_train.set_xlabel("Epoki")
    ax_train.set_ylabel(metric_label)
    ax_train.set_title(f"Treningowe {metric_label}")
    ax_train.set_ylim(bottom=0)
    ax_train.legend()
    ax_train.grid()

    ax_val.set_xlabel("Epoki")
    ax_val.set_ylabel(metric_label)
    ax_val.set_title(f"Walidacyjne {metric_label}")
    ax_val.set_ylim(bottom=0)
    ax_val.legend()
    ax_val.grid()

    plt.tight_layout()
    plt.show()
