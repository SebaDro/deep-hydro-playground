import matplotlib.pyplot as plt


def plot_grad_cam(timesteps, inputs, heatmap_timeseries_list, basins):
    fig, axis = plt.subplots(timesteps, 8, figsize=(16, 16), sharex="all", sharey="all")

    for t in range(0, timesteps):
        image = inputs[0, t, ..., 0]
        axis[t, 0].imshow(image, cmap="Blues")
        for i in range(0, 7):
            ax = axis[t, i + 1]
            ax.imshow(heatmap_timeseries_list[i][t], cmap="jet", alpha=0.4, vmin=0, vmax=1)

    for ax, col in zip(axis[0, 1:8], basins):
        ax.set_title(col)

    row_labels = [f"t={t}" for t in range(0, 11)]
    for ax, row in zip(axis[:, 0], row_labels):
        ax.set_ylabel(row)


def plot_grad_cam2(timesteps, inputs, heatmap_timeseries_list, basins):
    fig, axis = plt.subplots(timesteps, 8, figsize=(16, 16), sharex="all", sharey="all")

    for t in range(0, timesteps):
        image = inputs[0, t, ..., 0]
        axis[t, 0].imshow(image, cmap="Blues")
        for i in range(0, 7):
            ax = axis[t, i + 1]
            ax.imshow(heatmap_timeseries_list[t][i], cmap="jet", alpha=0.4)

    for ax, col in zip(axis[0, 1:8], basins):
        ax.set_title(col)

    row_labels = [f"t={t}" for t in range(0, 11)]
    for ax, row in zip(axis[:, 0], row_labels):
        ax.set_ylabel(row)