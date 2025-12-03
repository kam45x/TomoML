import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from dival import get_standard_dataset
from torch.utils.data import DataLoader


def interactive_window(image, a0=-200, b0=800):
    """
    Interaktywny podgląd obrazu CT z suwakiem w jednostkach Hounsfielda.

    Parametry:
    -----------
    image : np.ndarray
        Obraz 2D w zakresie 0.0-1.0
    a0 : float
        Początkowa minimalna wartość okna (HU)
    b0 : float
        Początkowa maksymalna wartość okna (HU)
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

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    im_display = ax.imshow(window_image(image, a0, b0), cmap="gray", vmin=0, vmax=1)

    ax_a = plt.axes([0.15, 0.1, 0.65, 0.03])
    ax_b = plt.axes([0.15, 0.05, 0.65, 0.03])

    slider_a = Slider(ax_a, "Min HU", hu_min, hu_max, valinit=a0)
    slider_b = Slider(ax_b, "Max HU", hu_min, hu_max, valinit=b0)

    def update(val):
        a_hu = slider_a.val
        b_hu = slider_b.val
        if b_hu <= a_hu:
            return
        im_display.set_data(window_image(image, a_hu, b_hu))
        fig.canvas.draw_idle()

    slider_a.on_changed(update)
    slider_b.on_changed(update)
    plt.show()


def main():
    dataset = get_standard_dataset(
        "custom",
        data_path="data/dataset_S2010",
        sinogram_shape=(512, 365),
        image_shape=(256, 256),
        parts_len={"train": 128, "validation": 16, "test": 17},
        impl="skimage",
    )

    test_dataset = dataset.create_torch_dataset(part="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
    )

    batch = next(iter(test_loader))

    image = batch[1][5].numpy()
    interactive_window(image, a0=-1024, b0=3071)


if __name__ == "__main__":
    main()
