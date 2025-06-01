import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import tifffile as tiff


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, show_images=False):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()

            if show_images:
                show_tensor_images(y_pred[0].cpu(), y[0].cpu(), x[0].cpu())
                break

    return total_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # Infer number of input channels from Sentinel image
    sample_input = tiff.imread(sentinel_list[0])
    if sample_input.ndim == 2:
        in_channels = 1
    elif sample_input.ndim == 3:
        in_channels = sample_input.shape[2]
    else:
        raise ValueError("Formato immagine Sentinel non supportato")

    # Infer number of output channels from Aerial image
    sample_output = tiff.imread(aerial_list[0])
    if sample_output.ndim == 2:
        out_channels = 1
    elif sample_output.ndim == 3:
        out_channels = sample_output.shape[2]
    else:
        raise ValueError("Formato immagine Aerea non supportato")

    print(f"Inferred input channels: {in_channels}")
    print(f"Inferred output channels: {out_channels}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(sentinel_list, aerial_list, batch_size=8)

    # Initialize model
    model = UNet(in_channels=in_channels, out_channels=out_channels).to(device)
    # model = vit.ViT(in_channels=in_channels, out_channels=out_channels, image_size=128).to(device)

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")

        val_loss = evaluate(model, val_loader, criterion, device, show_images=(epoch == epochs - 1))
        print(f"Validation Loss: {val_loss:.4f}")

    # Final test evaluation
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss (MSE): {test_loss:.4f}")

    # Show one example
    evaluate(model, test_loader, criterion, device, show_images=True)


if __name__ == "__main__":
    main()
