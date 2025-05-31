import torch
import matplotlib.pyplot as plt

def show_tensor_images(pred, target, sentinel=None):
    """
    Visualizza il target (immagine aerea reale), il pred (predetta) e opzionalmente l'input Sentinel.
    """
    def to_img(tensor):
        img = tensor.detach().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.transpose(1, 2, 0)

    fig, axs = plt.subplots(1, 3 if sentinel is not None else 2, figsize=(15, 5))
    axs[0].imshow(to_img(target))
    axs[0].set_title("Target")
    axs[0].axis("off")

    axs[1].imshow(to_img(pred))
    axs[1].set_title("Predicted")
    axs[1].axis("off")

    if sentinel is not None:
        axs[2].imshow(to_img(sentinel))
        axs[2].set_title("Sentinel (Input)")
        axs[2].axis("off")

    plt.tight_layout()
    plt.show()


def compute_mse(pred, target):
    return torch.mean((pred - target) ** 2).item()