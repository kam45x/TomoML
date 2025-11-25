import os
import pydicom
import matplotlib.pyplot as plt

# Configuration
ROOT_DIR = "C:\\Users\\kam45x\\OneDrive - Politechnika Warszawska\\przekroje"
OUTPUT_ACCEPT = "TomoML-main/src/ct_scan_accepted.txt"
OUTPUT_REJECT = "TomoML-main/src/ct_scan_rejected.txt"


def load_existing_categories():
    """Loads existing directories from category files."""
    accepted = set()
    rejected = set()

    if os.path.isfile(OUTPUT_ACCEPT):
        with open(OUTPUT_ACCEPT, "r") as fa:
            accepted = set(line.strip() for line in fa if line.strip())

    if os.path.isfile(OUTPUT_REJECT):
        with open(OUTPUT_REJECT, "r") as fr:
            rejected = set(line.strip() for line in fr if line.strip())

    return accepted, rejected


def show_dicom_info(dicom_path):
    """
    Reads and displays basic DICOM metadata and image.
    Returns True if the image should be rejected based on simple criteria.
    """
    print("\n=== DICOM FILE FOUND ===")
    print(f"File: {dicom_path}")

    ds = pydicom.dcmread(dicom_path)

    print(f"SOP Class UID = {ds.SOPClassUID.name}")
    print(f"Image type = {ds.ImageType}")
    print(f"Image size = {ds.Rows} x {ds.Columns}")
    print("Pixel Spacing =", getattr(ds, "PixelSpacing", "Not found"))
    print("Bits Stored =", getattr(ds, "BitsStored", "Not found"))
    print("Pixel Representation =", getattr(ds, "PixelRepresentation", "Not found"))
    print("Rescale Intercept =", getattr(ds, "RescaleIntercept", "Not found"))
    print("Rescale Slope =", getattr(ds, "RescaleSlope", "Not found"))
    print(f"Min pixel value = {ds.pixel_array.min()}")
    print(f"Max pixel value = {ds.pixel_array.max()}")

    if ds.ImageType[0] != "ORIGINAL":
        return True

    if ds.Rows != ds.Columns:
        return True

    pixel_array = ds.pixel_array

    # Display full image
    plt.imshow(pixel_array, cmap="gray")
    plt.title("DICOM Image")
    plt.colorbar(label="Pixel Intensity")
    plt.show()

    return False


def ask_user_for_category(directory):
    """Prompt user to choose A / R / S."""
    while True:
        answer = input(f"Accept (A) or Reject (R) for {directory}? ").strip().upper()
        if answer in ("A", "R"):
            return answer
        print("Invalid input. Use A or R.")


def write_to_file(decision, relative_root):
    """Writes the directory to the appropriate file based on user decision."""
    if decision == "A":
        with open(OUTPUT_ACCEPT, "a") as fa:
            fa.write(relative_root + "\n")
    elif decision == "R":
        with open(OUTPUT_REJECT, "a") as fr:
            fr.write(relative_root + "\n")


def main():
    accepted, rejected = load_existing_categories()

    print("Loaded existing categories:")
    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")
    print(f"Sum = {len(accepted) + len(rejected)} directories.\n")

    for root, dirs, files in os.walk(ROOT_DIR):
        if not dirs:  # only deepest subdirectories
            relative_root = os.path.relpath(root, ROOT_DIR)
            if relative_root in accepted or relative_root in rejected:
                continue

            dicom_file = None
            for f in files:
                if f == "I10" or f == "I10.dcm":
                    dicom_file = os.path.join(root, f)
                    break

            if dicom_file:
                reject = show_dicom_info(dicom_file)
                if reject:
                    print(f"Automatically rejecting {root}.")
                    write_to_file("R", relative_root)
                    continue

                decision = ask_user_for_category(relative_root)
                write_to_file(decision, relative_root)

                print(f"Recorded: {relative_root}, decision - {decision}\n")

    print("\n--- Finished scanning directories ---")


if __name__ == "__main__":
    main()
