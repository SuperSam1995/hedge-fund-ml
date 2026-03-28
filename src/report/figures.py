from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_cum(series_long: pd.DataFrame, strategy: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sub = (
        series_long[series_long["strategy"] == strategy]
        .pivot(index="date", columns="role", values="return")
        .sort_index()
    )
    cum = (1 + sub).cumprod()
    ax = cum.plot()
    ax.set_title(f"Cumulative return – {strategy}")
    ax.figure.tight_layout()
    ax.figure.savefig(outdir / f"cum_{strategy}.png", dpi=140)
    plt.close(ax.figure)


def plot_roll_te(series_long: pd.DataFrame, strategy: str, outdir: Path, win=24) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sub = (
        series_long[series_long["strategy"] == strategy]
        .pivot(index="date", columns="role", values="return")
        .sort_index()
    )
    te = (sub["target"] - sub["replica"]).rolling(win).std() * (12**0.5)
    ax = te.plot()
    ax.set_title(f"Rolling TE ({win}m, annualized) – {strategy}")
    ax.figure.tight_layout()
    ax.figure.savefig(outdir / f"rollTE_{strategy}.png", dpi=140)
    plt.close(ax.figure)
