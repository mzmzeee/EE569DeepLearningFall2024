import numpy as np
import matplotlib.pyplot as plt
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


def compute_covariance(data):
    """
    Compute the covariance matrix of the data.

    Parameters:
        data (array): Mean-centered data.

    Returns:
        array: Covariance matrix.
    """
    return np.cov(data, rowvar=False)


def perform_pca(cov_matrix, num_components=1):
    """
    Perform Principal Component Analysis (PCA) on the covariance matrix.

    Parameters:
        cov_matrix (array): Covariance matrix.
        num_components (int): Number of principal components to retain.

    Returns:
        tuple: Eigenvalues, eigenvectors, and the selected principal components.
    """
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    selected_components = eigenvectors[:, :num_components]
    return eigenvalues, eigenvectors, selected_components


def project_data(data, components):
    """
    Project the data onto the selected principal components.

    Parameters:
        data (array): Mean-centered data.
        components (array): Selected principal components.

    Returns:
        array: Projected data.
    """
    return np.dot(data, components)


def plot_original_data(data, mean, principal_component, projected_data):
    """
    Plot the original data, principal component, and projected data.

    Parameters:
        data (array): Original data.
        mean (array): Mean of the data.
        principal_component (array): Principal component vector.
        projected_data (array): Projected data points.
    """
    plt.figure(figsize=(8, 6))

    # Plot original data
    plt.scatter(data[:, 0], data[:, 1], label="Original Data", s=50, marker='o')

    # Plot principal component as a vector
    plt.quiver(*mean, *principal_component, color='r', scale=5, label="Principal Component 1")

    # Plot projection lines and projected data
    for i in range(len(data)):
        plt.plot([data[i, 0], projected_data[i, 0]], [data[i, 1], projected_data[i, 1]], 'g--')
    plt.scatter(projected_data[:, 0], projected_data[:, 1], label="Projected Data", s=50, marker='D', color='m')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Principal Component Analysis")
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    plt.gca().set_aspect('equal')  # Equal aspect ratio
    plt.show()


def plot_variance_explained(eigenvalues):
    """
    Plot the variance explained by each principal component.

    Parameters:
        eigenvalues (array): Eigenvalues of the covariance matrix.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(eigenvalues)), eigenvalues, label="Eigenvalues")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.title("Variance Explained by Principal Components")
    plt.xticks(range(len(eigenvalues)), [f"PC{i+1}" for i in range(len(eigenvalues))])
    plt.legend()
    plt.show()


# Main program
if __name__ == "__main__":
    # Configuration
    CLASS_SIZE = 20
    MEAN1 = np.array([1, 2])
    COV1 = np.array([[1, 0.5], [0.5, 5]])

    # Step 1: Generate data
    data = generate_data(MEAN1, COV1, CLASS_SIZE)

    # Step 2: Mean centering
    centered_data, mean = mean_center(data)

    # Step 3: Compute covariance matrix
    cov_matrix = compute_covariance(centered_data)

    # Step 4: Perform PCA
    eigenvalues, eigenvectors, selected_components = perform_pca(cov_matrix, num_components=1)

    # Step 5: Project data onto the first principal component
    reduced_data = project_data(centered_data, selected_components)

    # Reconstruct projected data in 2D for visualization
    projected_data = np.dot(reduced_data, selected_components.T) + mean

    # Output results
    print("Mean-centered data:\n", centered_data)
    print("Covariance matrix:\n", cov_matrix)
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)
    print("Selected principal component:\n", selected_components)
    print("Reduced data (1D):\n", reduced_data)

    # Step 6: Visualization
    plot_original_data(data, mean, selected_components[:, 0], projected_data)
    plot_variance_explained(eigenvalues)