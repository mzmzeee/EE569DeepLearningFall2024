import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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
    plt.title("VAE Reconstruction")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    plt.gca().set_aspect('equal')  # Equal aspect ratio
    plt.show()


# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for sampling from the latent distribution.

        Parameters:
            mu (tensor): Mean of the latent distribution.
            log_var (tensor): Log variance of the latent distribution.

        Returns:
            tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)  # Random noise with same size as std
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var


def train_VAE(model, data_loader, num_epochs=100, learning_rate=0.001):
    """Trains the VAE model.

    Args:
        model: The VAE model.
        data_loader: The data loader.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.

    Returns:
        A list of training losses.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []

    def loss_function(recon_x, x, mu, log_var):
        """VAE loss function."""
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_div

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_data in data_loader:
            inputs = batch_data[0]
            optimizer.zero_grad()
            reconstruction, mu, log_var = model(inputs)
            loss = loss_function(reconstruction, inputs, mu, log_var)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(data_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(data_loader):.4f}")

    return train_losses

# Main program
if __name__ == "__main__":
    # Configuration
    CLASS_SIZE = 50
    MEAN1 = np.array([1, 2])
    COV1 = np.array([[5, -1], [-1, 1]])
    batch_size = 32
    num_epochs = 1000
    latent_dim = 1
    hidden_dim = 32

    # Step 1: Generate data
    data = generate_data(MEAN1, COV1, CLASS_SIZE)

    # Step 2: Mean centering
    centered_data, mean = mean_center(data)

    # Convert data to PyTorch tensors
    centered_data_tensor = torch.tensor(centered_data, dtype=torch.float32)

    # Create a DataLoader for batching and shuffling
    dataset = TensorDataset(centered_data_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define model parameters
    input_dim = centered_data.shape[1]

    # Create the VAE model
    model = VAE(input_dim, latent_dim, hidden_dim)

    # Train the VAE
    train_losses = train_VAE(model, data_loader, num_epochs=num_epochs)

    # Get reconstructed data
    with torch.no_grad():
        reconstructed_data, _, _ = model(centered_data_tensor)
    reconstructed_data = reconstructed_data.numpy()
    reconstructed_data += mean

    # Step 6: Visualization
    plot_original_data(data, reconstructed_data)
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()