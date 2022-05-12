import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import textwrap


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


def plot_forcings(xds, variables: list, timesteps: int, cmaps: list, gpd_catchment = None, gpd_subbasins = None):
    fig, axis = plt.subplots(len(variables), timesteps, figsize=(10, 10), sharex="all", sharey="all")
    for i, var in enumerate(variables):
        min = np.nanmin(xds[var].values)
        max = np.nanmax(xds[var].values)
        levels = np.arange(min, max, 1)
        var_name = xds[var].attrs["long_name"]
        var_unit = xds[var].attrs["units"]
        for t in range(0, timesteps):
            if t == timesteps - 1:
                xds[var].isel(time=t).plot.contourf(ax=axis[i, t], add_labels=False, cmap=cmaps[i], levels=levels,
                                                    cbar_kwargs={"label": textwrap.fill(f"{var_name} [{var_unit}]", 20)})
            else:
                xds[var].isel(time=t).plot.contourf(ax=axis[i, t], add_labels=False, cmap=cmaps[i], levels=levels,
                                                        add_colorbar=False)
            if gpd_catchment is not None:
                gpd_catchment.plot(ax=axis[i, t], color='none', edgecolor='black')
            if gpd_subbasins is not None:
                gpd_subbasins.plot(ax=axis[i, t], color="none", linewidth=0.5, edgecolor='black')
    for i, ax in enumerate(axis[0]):
        time = xds.time.values[i]
        time_str = np.datetime_as_string(time, unit="D")
        ax.set_title(f"time={time_str}")
    plt.suptitle("DWD HYRAS timeseries for Wupper catchment area", fontsize=14)
    fig.supxlabel('longitude')
    fig.supylabel('latitude')
    plt.tight_layout()
    plt.show()


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