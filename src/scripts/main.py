from PatchDataset import create_dataloaders
import matplotlib.pyplot as plt
import torch

def show_tensor_image(sentinel_tensor, aerial_tensor):
    def normalize_img(tensor):
        img = tensor[:3].numpy()  # Prende solo i primi 3 canali
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.transpose(1, 2, 0)  # (H, W, C)

    sentinel_img = normalize_img(sentinel_tensor)
    aerial_img = normalize_img(aerial_tensor)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sentinel_img)
    axes[0].set_title("Sentinel Patch")
    axes[0].axis("off")

    axes[1].imshow(aerial_img)
    axes[1].set_title("Aerial Patch")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    sentinel_list = [
        "../../data/BRISIGHELLA/Sentinel/Sentinel2_post_Brisighella_2m.tif",
        "../../data/BRISIGHELLA/Sentinel/Sentinel2_pre_Brisighella_2m.tif",
        "../../data/CASOLA_VALSENIO/Sentinel/Sentinel2_post_Casola_2m.tif",
        "../../data/CASOLA_VALSENIO/Sentinel/Sentinel2_pre_Casola_2m.tif",
        "../../data/MODIGLIANA/Sentinel/Sentinel2_post_modigliana_2m.tif",
        "../../data/MODIGLIANA/Sentinel/Sentinel2_pre_modigliana_2m.tif",
        "../../data/PREDAPPIO/Sentinel/Sentinel2_post_Predappio_2m.tif",
        "../../data/PREDAPPIO/Sentinel/Sentinel2_pre_Predappio_2m.tif"
    ]

    aerial_list = [
        "../../data/BRISIGHELLA/Aerial/BRISIGHELLA_cgr_2023_2m.tif",
        "../../data/BRISIGHELLA/Aerial/BRISIGHELLA_agea_2020_2m.tif",
        "../../data/CASOLA_VALSENIO/Aerial/CASOLA_VALSENIO_cgr_2023_2m.tif",
        "../../data/CASOLA_VALSENIO/Aerial/CASOLA_VALSENIO_agea_2020_2m.tif",
        "../../data/MODIGLIANA/Aerial/MODIGLIANA_cgr_2023_2m.tif",
        "../../data/MODIGLIANA/Aerial/MODIGLIANA_agea_2020_2m.tif",
        "../../data/PREDAPPIO/Aerial/PREDAPPIO_cgr_2023_2m.tif",
        "../../data/PREDAPPIO/Aerial/PREDAPPIO_agea_2020_2m.tif"
    ]

    train_loader, val_loader, test_loader = create_dataloaders(
        sentinel_list,
        aerial_list,
        patch_size=128,
        stride=64,
        augment=True,
        normalize=True,
        batch_size=32
    )

    for batch in train_loader:
        x_batch, y_batch = batch
        show_tensor_image(x_batch[0], y_batch[0])
        break


if __name__ == '__main__':
    main()