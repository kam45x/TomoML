import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from dival import get_standard_dataset
import torch
from torch.utils.data import DataLoader
import astra

from models.Pix2Pix_128 import UnetGenerator


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
    ax1.set_title("Original")

    im2 = ax2.imshow(window_image(image2, a0, b0), cmap="gray", vmin=0, vmax=1)
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

            return (img.cpu().numpy().astype(np.float32), output.cpu().numpy().astype(np.float32))


def get_reconstructed_images_classic(
    sinograms: np.ndarray, img_size, algorithm="FBP_CUDA"
):
    """
    Rekonstruuje batch obrazów z batcha sinogramów.

    Parametry:
        sinograms (np.ndarray): [B, A, D]
        num_angles (int): liczba kątów
        img_size (int): rozmiar obrazu wyjściowego
        algorithm (str): np. "FBP" lub "FBP_CUDA"

    Zwraca:
        np.ndarray: [B, img_size, img_size]
    """

    assert sinograms.ndim == 3, "Oczekiwany shape [B, A, D]"
    B, A, D = sinograms.shape

    # --- Geometria (wspólna dla batcha) ---
    angles = np.linspace(0, np.pi, A, endpoint=False)
    proj_geom = astra.create_proj_geom("parallel", 1.0, D, angles)
    vol_geom = astra.create_vol_geom(img_size, img_size)

    reconstructions = []

    for b in range(B):
        sino = sinograms[b]

        sino_id = astra.data2d.create("-sino", proj_geom, sino)
        reco_id = astra.data2d.create("-vol", vol_geom)

        cfg = astra.astra_dict(algorithm)
        cfg["ReconstructionDataId"] = reco_id
        cfg["ProjectionDataId"] = sino_id
        if "CUDA" not in algorithm:
            cfg["ProjectorId"] = astra.create_projector("linear", proj_geom, vol_geom)

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations=256 * 500)

        reco = astra.data2d.get(reco_id)
        reconstructions.append(reco)

        # Sprzątanie
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sino_id)
        astra.data2d.delete(reco_id)

    return np.stack(reconstructions, axis=0)


def main():
    RECONSTRUCTOR = "FBP_CUDA"
    index = 0

    dataset = get_standard_dataset(
        "custom",
        data_path="../data/ct_reconstruction_dataset_128",
        sinogram_shape=(256, 183),
        image_shape=(128, 128),
        parts_len={"train": 206143, "validation": 25767, "test": 25769},
        impl="skimage",
    )

    test_dataset = dataset.create_torch_dataset(part="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
    )

    if RECONSTRUCTOR == "NN":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        generator = UnetGenerator().to(device)
        weights = torch.load("../models/128x128_1412_pix2pix.pth", map_location=device)
        generator.load_state_dict(weights)
        original, reconstructed = get_reconstructed_images_nn(generator, test_loader, device)
        img_original = original[index][0]
        img_reconstructed = reconstructed[index][0]
    else:
        with torch.no_grad():
            for sino, img in test_loader:
                original = img.unsqueeze(1).cpu().numpy().astype(np.float32)
                original[index][0] = np.rot90(original[index][0])
                sino_np = sino.detach().cpu().numpy().astype(np.float32)

                reconstructed = get_reconstructed_images_classic(
                    sino_np, img_size=128, algorithm=RECONSTRUCTOR
                )
                reconstructed = reconstructed * 512
                img_original = original[index][0]
                img_reconstructed = reconstructed[index]
                break
    
    slider_window(img_original, img_reconstructed, a0=-1024, b0=3071)


if __name__ == "__main__":
    main()
