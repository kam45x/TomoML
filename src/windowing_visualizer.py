import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from dival import get_standard_dataset
from dival.reconstructors.odl_reconstructors import FBPReconstructor
import odl
import torch
from torch.utils.data import DataLoader

from models.Pix2Pix_128 import UnetGenerator


def slider_window(image1, image2, a0=-1024, b0=3071):
    """
    Wyswietla dwa obrazy CT jednoczesnie (np. oryginalny + rekonstrukcje)
    z jednym wspolnym zestawem suwakow w jednostkach HU.

    Parametry:
    -----------
    image1 : np.ndarray
        Pierwszy obraz 2D w zakresie 0.0-1.0
    image2 : np.ndarray
        Drugi obraz 2D w zakresie 0.0-1.0
    a0, b0 : float
        Poczatkowe wartosci okienka HU
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

    im1 = ax1.imshow(
        window_image(image1, a0, b0), cmap="gray", vmin=0, vmax=1
    )
    ax1.set_title("Original")

    im2 = ax2.imshow(
        window_image(image2, a0, b0), cmap="gray", vmin=0, vmax=1
    )
    ax2.set_title("Reconstructed")

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


def get_reconstructed_images_nn(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for sino, img in dataloader:
            sino = sino.unsqueeze(1).to(device, non_blocking=True)
            img = img.unsqueeze(1).to(device, non_blocking=True)

            output = model(sino)

            return (
                img.cpu().numpy().astype(np.float32),
                output.cpu().numpy().astype(np.float32),
            )


def build_fbp_reconstructor(
    img_size=128, num_angles=256, impl="astra_cuda"
):
    """
    Buduje rekonstruktor FBP przy uzyciu ODL (tak jak w test_model.ipynb).

    Parametry:
        img_size   : rozmiar obrazu wyjsciowego (kwadrat)
        num_angles : liczba katow projekcji
        impl       : backend ODL ("astra_cuda" lub "skimage")

    Zwraca:
        FBPReconstructor gotowy do uzycia przez .reconstruct(observation)
    """
    reco_space = odl.uniform_discr(
        min_pt=[-0.13, -0.13],
        max_pt=[0.13, 0.13],
        shape=(img_size, img_size),
        dtype=np.float32,
    )
    reco_geometry = odl.tomo.parallel_beam_geometry(
        reco_space, num_angles=num_angles
    )
    reco_ray_trafo = odl.tomo.RayTransform(
        reco_space, reco_geometry, impl=impl
    )

    return FBPReconstructor(reco_ray_trafo)


def get_reconstructed_images_classic(
    dataset, index=0, impl="astra_cuda"
):
    """
    Rekonstruuje obraz metoda FBP (ODL) dla wybranego elementu ze zbioru testowego.

    Parametry:
        dataset : dival dataset z metoda get_data_pairs
        index   : indeks probki w zbiorze testowym
        impl    : backend ODL ray transform ("astra_cuda" lub "skimage")

    Zwraca:
        tuple (img_original, img_reconstructed) - oba jako np.ndarray (128, 128)
    """
    test_pairs = dataset.get_data_pairs(part="test")
    observation, ground_truth = test_pairs[index]

    reconstructor = build_fbp_reconstructor(
        img_size=128, num_angles=256, impl=impl
    )
    reconstruction = np.asarray(
        reconstructor.reconstruct(observation)
    )
    original = np.asarray(ground_truth)

    return original, reconstruction


def main():
    RECONSTRUCTOR = "FBP"  # "FBP", "FBP_CUDA", or "NN"
    index = 0

    dataset = get_standard_dataset(
        "custom",
        data_path="../data/ct_reconstruction_dataset_128",
        sinogram_shape=(256, 183),
        image_shape=(128, 128),
        parts_len={
            "train": 206143,
            "validation": 25767,
            "test": 25769,
        },
        impl="skimage",
    )

    if RECONSTRUCTOR == "NN":
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        generator = UnetGenerator().to(device)
        weights = torch.load(
            "../models/128x128_1412_pix2pix.pth", map_location=device
        )
        generator.load_state_dict(weights)

        test_dataset = dataset.create_torch_dataset(part="test")
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=True
        )

        original, reconstructed = get_reconstructed_images_nn(
            generator, test_loader, device
        )
        img_original = original[index][0]
        img_reconstructed = reconstructed[index][0]
    else:
        # "FBP_CUDA" uses astra_cuda backend; "FBP" falls back to skimage
        impl = (
            "astra_cuda" if RECONSTRUCTOR == "FBP_CUDA" else "skimage"
        )
        img_original, img_reconstructed = (
            get_reconstructed_images_classic(
                dataset, index=index, impl=impl
            )
        )

    slider_window(img_original, img_reconstructed, a0=-1024, b0=3071)


if __name__ == "__main__":
    main()
