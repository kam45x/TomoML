import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from dival import get_standard_dataset
import torch
from torch.utils.data import DataLoader

from models.Pix2Pix_256_V1 import UnetGenerator


def slider_window(image1, image2, a0=-1024, b0=3071):
    """
    Wyświetla dwa obrazy CT jednocześnie (np. oryginalny + rekonstrukcję)
    z jednym wspólnym zestawem suwaków w jednostkach HU.

    Parametry:
    -----------
    image1 : np.ndarray
        Pierwszy obraz 2D w zakresie 0.0–1.0
    image2 : np.ndarray
        Drugi obraz 2D w zakresie 0.0–1.0
    a0, b0 : float
        Początkowe wartości okienka HU
    """

    hu_min = -1024
    hu_max = 3071

    def window_image(img, a_hu, b_hu):
        # konwersja HU do skali 0-1
        a = (a_hu - hu_min) / (hu_max - hu_min)
        b = (b_hu - hu_min) / (hu_max - hu_min)
        img_windowed = np.clip(img, a, b)
        img_windowed = (img_windowed - a) / (b - a)
        return img_windowed

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.25)

    ax1, ax2 = axes

    im1 = ax1.imshow(window_image(image1, a0, b0), cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Obraz 1")

    im2 = ax2.imshow(window_image(image2, a0, b0), cmap="gray", vmin=0, vmax=1)
    ax2.set_title("Obraz 2")

    ax_a = plt.axes([0.15, 0.1, 0.65, 0.03])
    ax_b = plt.axes([0.15, 0.05, 0.65, 0.03])

    slider_a = Slider(ax_a, "Min HU", hu_min, hu_max, valinit=a0)
    slider_b = Slider(ax_b, "Max HU", hu_min, hu_max, valinit=b0)

    def update(val):
        a_hu = slider_a.val
        b_hu = slider_b.val
        if b_hu <= a_hu:
            return
        im1.set_data(window_image(image1, a_hu, b_hu))
        im2.set_data(window_image(image2, a_hu, b_hu))
        fig.canvas.draw_idle()

    slider_a.on_changed(update)
    slider_b.on_changed(update)

    plt.show()


def get_reconstructed_images(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for sino, img in dataloader:
            sino = sino.unsqueeze(1).to(device, non_blocking=True)
            img = img.unsqueeze(1).to(device, non_blocking=True)

            output = model(sino)

            return (img, output)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    generator = UnetGenerator().to(device)
    weights = torch.load("models/256x256_0212_pix2pix.pth", map_location=device)
    generator.load_state_dict(weights)

    dataset = get_standard_dataset(
        "custom",
        data_path="data/dataset_S2010",
        sinogram_shape=(512, 365),
        image_shape=(256, 256),
        parts_len={"train": 128, "validation": 16, "test": 17},
        impl="skimage",
    )

    test_dataset = dataset.create_torch_dataset(part="validation")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
    )

    original, reconstructed = get_reconstructed_images(generator, test_loader, device)

    index = 0
    img_original = original[index][0].cpu().numpy()
    img_reconstructed = reconstructed[index][0].cpu().numpy()
    
    slider_window(img_original, img_reconstructed, a0=-1024, b0=3071)


if __name__ == "__main__":
    main()
