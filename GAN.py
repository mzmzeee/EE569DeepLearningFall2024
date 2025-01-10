import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt

def generate_data(num_samples, means, stds):
  """
  Generates data points from a mixture of Gaussians.

  Args:
      num_samples: Total number of data points to generate.
      means: List of means for each Gaussian distribution.
      stds: List of standard deviations for each Gaussian distribution.

  Returns:
      A NumPy array of shape (num_samples, 2) containing the generated data points.
  """
  data = []
  for mean, std in zip(means, stds):
    data.extend(np.random.normal(loc=mean, scale=std, size=(num_samples // len(means), 2)))
  return np.array(data)

# Models
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(generator, discriminator, data, latent_dim, epochs, batch_size, lr):
  """
  Trains a GAN with a specified mode dropping factor to discourage mode collapse.

  Args:
      generator: Generator network instance.
      discriminator: Discriminator network instance.
      data: Training data as a NumPy array.
      latent_dim: Dimensionality of the latent space.
      epochs: Number of training epochs.
      batch_size: Batch size for training.
      lr: Learning rate for optimizers.

  Returns:
      A tuple containing lists of generator and discriminator losses, and a list of generated samples history.
  """
  criterion = nn.BCELoss()
  g_optimizer = Adam(generator.parameters(), lr=lr)
  d_optimizer = Adam(discriminator.parameters(), lr=lr)

  real_labels = torch.full((batch_size, 1), 0.9)
  fake_labels = torch.full((batch_size, 1), 0.1)

  losses_g = []
  losses_d = []
  generated_samples_history = []

  for epoch in range(epochs):
      for i in range(0, len(data), batch_size):
          real_batch = torch.tensor(data[i:i + batch_size], dtype=torch.float32)

          # Train Discriminator
          d_optimizer.zero_grad()
          real_outputs = discriminator(real_batch)
          d_real_loss = criterion(real_outputs, real_labels[:len(real_batch)])

          z = torch.randn(len(real_batch), latent_dim)
          fake_batch = generator(z)
          fake_outputs = discriminator(fake_batch.detach())  # Detach to prevent generator updates
          d_fake_loss = criterion(fake_outputs, fake_labels[:len(fake_batch)])
          d_loss = d_real_loss + d_fake_loss
          d_loss.backward()
          d_optimizer.step()

          # Train Generator
          g_optimizer.zero_grad()
          z = torch.randn(len(real_batch), latent_dim)
          fake_batch = generator(z)
          fake_outputs = discriminator(fake_batch)
          g_loss = criterion(fake_outputs, real_labels[:len(fake_batch)])
          g_loss.backward()
          g_optimizer.step()

      losses_g.append(g_loss.item())
      losses_d.append(d_loss.item())

      # Store generated samples for visualization (every few epochs)
      if (epoch + 1) % max(1, epochs // 12) == 0:  # ensure it always runs at least once
          with torch.no_grad():
              z = torch.randn(1024, latent_dim)
              generated_samples = generator(z).numpy()
              generated_samples_history.append(generated_samples)

      print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
  return losses_g, losses_d, generated_samples_history


def plot_results(data, generated_samples_history, losses_g, losses_d):
    """Plots the generated samples and training losses."""
    num_plots = len(generated_samples_history)
    plt.figure(figsize=(
    15, 5 * (num_plots // 4 + (1 if num_plots % 4 != 0 else 0))))  # adjust figure size based on number of plots
    for i, samples in enumerate(generated_samples_history):
        plt.subplot(num_plots // 4 + (1 if num_plots % 4 != 0 else 0), 4, i + 1)
        plt.scatter(data[:, 0], data[:, 1], label="Real Data", alpha=0.3, s=5)
        plt.scatter(samples[:, 0], samples[:, 1], label="Generated Data", alpha=0.5, color='red', s=5)
        plt.title(f"Epoch {(i + 1) * (epochs // len(generated_samples_history))}")  # correct epoch number
        plt.gca().set_aspect('equal')
        plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot losses
    plt.figure(figsize=(8, 6))
    plt.plot(losses_g, label="Generator Loss")
    plt.plot(losses_d, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training Losses")
    plt.show()


# Main
if __name__ == "__main__":
    # Hyperparameters
    latent_dim = 2
    output_dim = 2
    batch_size = 512
    epochs = 1000
    lr = 0.001

    # Data
    dis = 10
    means = [[-dis, 0], [dis, 0], [0, -dis], [0, dis],
             [-dis/1.41, -dis/1.41], [dis/1.41, dis/1.41], [-dis/1.41, dis/1.41], [dis/1.41, -dis/1.41]]
    stds = [0.3] * len(means)  # Reduced std for tighter clusters
    num_samples = 2048
    data = generate_data(num_samples, means, stds)

    # Models
    generator = Generator(latent_dim, output_dim)
    discriminator = Discriminator(output_dim)

    # Training
    losses_g, losses_d, generated_samples_history = train_gan(generator, discriminator, data, latent_dim, epochs,
                                                              batch_size, lr)
    plot_results(data, generated_samples_history, losses_g, losses_d)
