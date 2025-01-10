import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from scipy.stats import multivariate_normal


def generate_data(mean, cov, size):
    """
    Generate random data from a multivariate normal distribution.

    Parameters:
        mean (array): Mean of the distribution.
        cov (array): Covariance matrix.
        size (int): Number of data points to generate.

    Returns:
        array: Generated data points.
    """
    return multivariate_normal.rvs(mean, cov, size)


def mean_center(data):
    """
    Perform mean centering on the data.

    Parameters:
        data (array): Original data.

    Returns:
        tuple: Mean-centered data and the mean.
    """
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    return centered_data, mean


def plot_original_data(data, reconstructed_data):
    """
    Plot the original data and reconstructed data.

    Parameters:
        data (array): Original data.
        reconstructed_data (array): Reconstructed data.
    """
    plt.figure(figsize=(8, 6))

    # Plot original data
    plt.scatter(data[:, 0], data[:, 1], label="Original Data", s=50, marker='o')

    # Plot reconstructed data
    plt.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], label="Reconstructed Data", s=50, marker='x', color='red')

    for i in range(len(data)):
        plt.plot([data[i, 0], reconstructed_data[i, 0]], [data[i, 1], reconstructed_data[i, 1]], 'g--')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Autoencoder Reconstruction")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    plt.gca().set_aspect('equal')  # Equal aspect ratio
    plt.show()


def plot_mnist_digits(images):
    """
    Plot a grid of MNIST digits.

    Parameters:
        images (array): MNIST images to plot.
    """
    rows, cols = 4, 4
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(images[i * cols + j], cmap="gray")
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    plt.tight_layout()
    plt.show()


# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Sigmoid activation for output (data between 0-1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

    def decode(self, code):
        decoded = self.decoder(code)
        return decoded




def train_autoencoder(model, data_loader, num_epochs=100, learning_rate=0.001):
    """Trains the autoencoder model.

    Args:
        model: The autoencoder model.
        data_loader: The data loader.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.

    Returns:
        A list of training losses.
    """
    criterion = nn.MSELoss()  # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_data in data_loader:
            inputs = batch_data[0].reshape(-1, input_dim) # Get the input batch
            optimizer.zero_grad()  # Zero the gradients
            outputs,_ = model(inputs)  # Forward pass
            loss = criterion(outputs, inputs)  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(data_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(data_loader):.4f}")
    return train_losses


# Main program
if __name__ == "__main__":
    # Configuration
    use_mnist = True  # Set to True to use MNIST, False for generated data
    batch_size = 32
    num_epochs = 20
    hidden_dim = 64
    latent_dim = 10

    if use_mnist:
        # Load MNIST dataset
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )
        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )

        # Create data loaders
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        # Get a sample batch and reshape it to (batch_size, 784)
        sample_batch = next(iter(train_dataloader))[0]
        input_dim = sample_batch.shape[1] * sample_batch.shape[2] * sample_batch.shape[3] # 28*28 = 784
        model = Autoencoder(input_dim, latent_dim, hidden_dim)

        # Train the autoencoder
        train_losses = train_autoencoder(model, train_dataloader, num_epochs=num_epochs)

        # Get reconstructed data
        with torch.no_grad():
            sample_batch = sample_batch.reshape(-1, input_dim)
            reconstructed_batch,_ = model(sample_batch)
            reconstructed_images = reconstructed_batch.reshape(-1, 28, 28).numpy()
            original_images = sample_batch.reshape(-1, 28, 28).numpy()

        plot_mnist_digits(original_images)
        plot_mnist_digits(reconstructed_images)
        plt.plot(train_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()

        # Get reconstructed data
        with torch.no_grad():
            sample_batch = sample_batch.reshape(-1, input_dim)
            _, codes = model(sample_batch)
            reconstructed_batch = model.decode(codes+10*torch.randn_like(codes))
            reconstructed_images = reconstructed_batch.reshape(-1, 28, 28).numpy()
            original_images = sample_batch.reshape(-1, 28, 28).numpy()

        plot_mnist_digits(original_images)
        plot_mnist_digits(reconstructed_images)

    else:
        # Configuration for generated data
        CLASS_SIZE = 100
        MEAN1 = np.array([1, 2])
        COV1 = np.array([[1, 1], [1, 1]])
        input_dim = 2

        # Step 1: Generate data
        data = generate_data(MEAN1, COV1, CLASS_SIZE)

        # Step 2: Mean centering
        centered_data, mean = mean_center(data)

        # Convert data to PyTorch tensors
        centered_data_tensor = torch.tensor(centered_data, dtype=torch.float32)

        # Create a DataLoader for batching and shuffling
        dataset = TensorDataset(centered_data_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create the autoencoder model
        model = Autoencoder(input_dim, latent_dim, hidden_dim)

        # Train the autoencoder
        train_losses = train_autoencoder(model, data_loader, num_epochs=num_epochs)

        # Get reconstructed data
        with torch.no_grad():
            reconstructed_data = model(centered_data_tensor).numpy()

        reconstructed_data += mean #add mean back

        # Step 6: Visualization
        plot_original_data(data, reconstructed_data)
        plt.plot(train_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()
