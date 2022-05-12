import matplotlib.pyplot as plt
import pandas as pd


def load_wv_streamflow(path: str):
    pd_streamflow = pd.read_csv(path, sep=",", skiprows=2, header=0, decimal=".")
    pd_streamflow["date"] = pd.to_datetime(pd_streamflow["date"])
    pd_streamflow = pd_streamflow.rename(columns={"date": "time"})
    pd_streamflow = pd_streamflow.melt(id_vars=["time"], var_name="basin", value_name="streamflow")
    pd_streamflow = pd_streamflow.set_index(["time", "basin"])
    ds_streamflow = pd_streamflow.to_xarray()
    return ds_streamflow


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