# %%
"""
Plotting utilities to visualize training logs.
"""
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

from pathlib import Path, PurePath
from IPython.display import clear_output


def plot_logs(
    logs, fields=("class_error", "loss_bbox_unscaled"), ewm_col=0, log_name="log.txt", slc=slice(None)
):
    """
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    """
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(
                f"{func_name} info: logs param expects a list argument, converted to list[Path]."
            )
        else:
            raise ValueError(
                f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}"
            )

    # verify valid dir(s) and that every item in list is Path object
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(
                f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}"
            )
        if dir.exists():
            continue
        raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True)[slc] for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == "mAP":
                coco_eval = (
                    pd.DataFrame(
                        pd.np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                    )
                    .ewm(com=ewm_col)
                    .mean()
                )
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    # y=[f"train_{field}"],
                    y=[f"train_{field}", f"val_{field}"],
                    ax=axs[j],
                    color=[color] * 2,
                    style=["-", "--"],
                )
    for ax, field in zip(axs, fields):
        names = []
        for name in [
            Path(p).name
            if Path(p).name != "log"
            else os.path.basename(os.path.dirname(str(p)))
            for p in logs
        ]:
            names.append("_".join(name.split("_")[-3:]) + "_train")
            names.append("_".join(name.split("_")[-3:]) + "_val")

        ax.legend(names)

        ax.set_title(field)


def plot_precision_recall(files, naming_scheme="iter"):
    if naming_scheme == "exp_id":
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == "iter":
        names = [f.stem for f in files]
    else:
        raise ValueError(f"not supported {naming_scheme}")
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(
        files, sns.color_palette("Blues", n_colors=len(files)), names
    ):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data["precision"]
        recall = data["params"].recThrs
        scores = data["scores"]
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data["recall"][0, :, 0, -1].mean()
        print(
            f"{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, "
            + f"score={scores.mean():0.3f}, "
            + f"f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}"
        )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title("Precision / Recall")
    axs[0].legend(names)
    axs[1].set_title("Scores / Recall")
    axs[1].legend(names)
    return fig, axs


# %%
logs = [
    # Path("/home/lyz/TT_outputs/tl_2/log"),
    Path("/home/lyz/TT_outputs/tl_9_rpos/log"),
    Path("/home/lyz/TT_outputs/tl_9/log"),
    Path("/home/lyz/TT_outputs/tl_17/log"),
]

ewm_col = 2

# slc = slice(3000, 6000)
# slc = slice(None, 500)
slc = slice(None)

while True:
    try:
        plot_logs(logs, fields=("loss", "loss_bce_unscaled"), ewm_col=ewm_col, slc=slc)
        plot_logs(logs, fields=("loss_bbox_unscaled", "loss_giou_unscaled"), ewm_col=ewm_col, slc=slc)
        # plot_logs(logs, fields=("pos_prop", "pos_prop"), ewm_col=ewm_col, slc=slc)
        plt.show()
    except ValueError:
        print("Fail")
        continue
    # break
    time.sleep(20)
    clear_output()


# plot_logs(
#     logs, fields=("loss_ce", "loss_giou"),
# )



# %%
