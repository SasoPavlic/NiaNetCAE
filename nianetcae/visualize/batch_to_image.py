import random

import numpy as np
import torch
from matplotlib import pyplot as plt


def normalize_and_rescale(tensor):
    # Convert tensor to NumPy array
    tensor_np = tensor.numpy()

    # Normalize values between 0 and 1
    normalized = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min())

    # Rescale values between 0 and 255
    rescaled = (normalized * 255).astype(np.uint8)

    # Expand dimensions if necessary
    if len(rescaled.shape) == 2:
        rescaled = np.expand_dims(rescaled, axis=2)

    return torch.from_numpy(rescaled)


def visualise_batch(fig_path, batch_idx, **kwargs):
    input = kwargs['input'].cpu()
    depth = kwargs['depth'].cpu()
    output = kwargs['output'].cpu()

    # Check if the batch sizes for image and depth match
    if input.shape[0] != depth.shape[0] != output.shape[0]:
        raise ValueError("Number of images and depths in the batch don't match.")

    # Determine the number of images in the batch and calculate the grid dimensions
    batch_size = input.shape[0]
    num_cols = 3  # Displaying image and depth side by side in one row
    num_rows = batch_size

    # Adjust the number of rows and columns if the last row is not fully occupied
    # if num_rows * num_cols < batch_size:
    #     num_rows += 1

    # Create a figure with subplots arranged in a grid
    fig_width = 12
    fig_height = 4 * batch_size
    # When batch size is 1, create a single subplot directly
    if batch_size == 1:
        fig, axs = plt.subplots(1, num_cols, figsize=(fig_width, fig_height))

        # Flatten the subplot array to simplify indexing
        axs = axs.flatten()

        image = input[0]
        image = np.transpose(image, (1, 2, 0))
        depth_map = depth[0]
        depth_map = np.transpose(depth_map, (1, 2, 0))
        reconstructed = output[0].detach()
        reconstructed = np.transpose(reconstructed, (1, 2, 0))

        image = normalize_and_rescale(image)
        depth_map = normalize_and_rescale(depth_map)
        reconstructed = normalize_and_rescale(reconstructed)

        # Display the image in the first column
        axs[0].imshow(image)
        axs[0].set_title('Image')
        axs[0].axis('off')

        # Display the depth in the second column
        axs[1].imshow(depth_map)
        axs[1].set_title('Depth')
        axs[1].axis('off')

        # Display the output in the third column
        axs[2].imshow(reconstructed)
        axs[2].set_title('Output')
        axs[2].axis('off')

        # Hide any remaining empty subplots
        for j in range(num_cols, len(axs)):
            axs[j].axis('off')

    else:
        fig, axs = plt.subplots(batch_size, num_cols, figsize=(fig_width, fig_height))

        # Iterate over the images and depths in the batch and display them in the subplots
        for i in range(batch_size):
            image = input[i]
            image = np.transpose(image, (1, 2, 0))
            depth_map = depth[i]
            depth_map = np.transpose(depth_map, (1, 2, 0))
            reconstructed = output[i].detach()
            reconstructed = np.transpose(reconstructed, (1, 2, 0))

            image = normalize_and_rescale(image)
            depth_map = normalize_and_rescale(depth_map)
            reconstructed = normalize_and_rescale(reconstructed)

            # Display the image in the first column
            axs[i][0].imshow(image)
            axs[i][0].set_title('Image')
            axs[i][0].axis('off')

            # Display the depth in the second column
            axs[i][1].imshow(depth_map)
            axs[i][1].set_title('Depth')
            axs[i][1].axis('off')

            # Display the output in the third column
            axs[i][2].imshow(reconstructed)
            axs[i][2].set_title('Output')
            axs[i][2].axis('off')

        # Hide any remaining empty subplots
        for j in range(batch_size * num_cols, len(axs.flatten())):
            axs.flatten()[j].axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{fig_path}_{batch_idx}_{random.randint(0, 1000)}.png')

    # Show the figure
    # #plt.show()
    # pass
