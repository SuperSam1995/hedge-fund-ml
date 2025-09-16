"""Data preparation pipeline extracted from notebooks."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from hedge_fund_ml.data import DataRegistry

logger = logging.getLogger(__name__)


class RegistrySettings(BaseModel):
    """Location of the :class:`~hedge_fund_ml.data.registry.DataRegistry`."""

    root: Path
    config: Path

    model_config = {"extra": "forbid"}

    def resolve(self) -> DataRegistry:
        return DataRegistry.from_yaml(self.root, self.config)


class SourceConfig(BaseModel):
    """Identifiers of datasets managed by the data registry."""

    risk_free: str
    navror: str
    etf: str

    model_config = {"extra": "forbid"}


class ProcessingConfig(BaseModel):
    """Processing window and transformation options."""

    start: date
    end: date
    frequency: str = Field(default="M")
    drop_etf_symbols: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @property
    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.end)

    @property
    def pandas_frequency(self) -> str:
        """Return a pandas-compatible frequency alias."""

        return "ME" if self.frequency == "M" else self.frequency


class OutputConfig(BaseModel):
    """Filesystem layout for prepared artefacts."""

    processed_dir: Path
    hedge_funds: Path
    factor_etf: Path
    risk_free: Path
    hedge_fund_names: Path
    factor_etf_names: Path
    mirrors: list[Path] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    def _resolve(self, base: Path, target: Path) -> Path:
        return target if target.is_absolute() else base / target

    def dataset_paths(self) -> Dict[str, Path]:
        base = self.processed_dir
        return {
            "hedge_funds": self._resolve(base, self.hedge_funds),
            "factor_etf": self._resolve(base, self.factor_etf),
            "risk_free": self._resolve(base, self.risk_free),
        }

    def metadata_paths(self) -> Dict[str, Path]:
        base = self.processed_dir
        return {
            "hedge_fund_names": self._resolve(base, self.hedge_fund_names),
            "factor_etf_names": self._resolve(base, self.factor_etf_names),
        }

    def mirror_dirs(self) -> Iterable[Path]:
        for target in self.mirrors:
            yield target if target.is_absolute() else Path.cwd() / target


class DataPrepConfig(BaseModel):
    """Top-level schema for the data preparation pipeline."""

    registry: RegistrySettings
    sources: SourceConfig
    processing: ProcessingConfig
    outputs: OutputConfig

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DataPrepConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.model_validate(payload)


@dataclass
class RawData:
    risk_free: pd.DataFrame
    hedge_fund: pd.DataFrame
    hedge_fund_names: Dict[str, str]
    etf_raw: pd.DataFrame
    etf_names: Dict[str, str]


@dataclass
class PreparedData:
    risk_free: pd.DataFrame
    hedge_funds: pd.DataFrame
    factor_etf: pd.DataFrame
    hedge_fund_names: Dict[str, str]
    factor_etf_names: Dict[str, str]


def _normalize_hfd_label(label: str) -> str:
    cleaned = label.strip()
    cleaned = cleaned.removeprefix("Credit Suisse ")
    cleaned = cleaned.removesuffix(" Hedge Fund Index")
    return " ".join(cleaned.split())


def _normalize_etf_label(label: str) -> str:
    cleaned = label or ""
    replacements = [
        ("Cboe ", ""),
        (" Index", ""),
        (" Total Return", ""),
        (" Total Return Value Unhedged USD", ""),
        (" Value Unhedged USD", ""),
        (".1", ""),
    ]
    for src, dest in replacements:
        cleaned = cleaned.replace(src, dest)
    return " ".join(cleaned.split())


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            series.loc[missing], format="%d-%m-%Y", errors="coerce"
        )
    return parsed


def load_raw(config: DataPrepConfig, registry: DataRegistry | None = None) -> RawData:
    """Load raw datasets defined in the registry."""

    registry = registry or config.registry.resolve()
    logger.info("Loading raw datasets via registry %s", config.registry.config)

    risk_free_path = registry.path(config.sources.risk_free)
    nav_path = registry.path(config.sources.navror)
    etf_path = registry.path(config.sources.etf)

    for name in (config.sources.risk_free, config.sources.navror, config.sources.etf):
        if registry.verify(name):
            logger.info("Verified checksum for %s", name)
        else:
            logger.warning("Checksum mismatch or missing checksum for %s", name)

    risk_free = pd.read_csv(risk_free_path, usecols=["Date", "RF"])

    nav_header = pd.read_csv(nav_path, nrows=1)
    nav_header = nav_header.drop(columns=["Unnamed: 0"], errors="ignore")
    hedge_fund_names: Dict[str, str] = {}
    for column, code in nav_header.iloc[0].items():
        code_str = str(code)
        if code_str == "Date":
            continue
        hedge_fund_names[code_str] = _normalize_hfd_label(str(column))

    hedge_fund = pd.read_csv(nav_path, skiprows=1)

    etf_raw = pd.read_csv(etf_path, header=None)
    name_row = etf_raw.iloc[0].fillna("")
    ticker_row = etf_raw.iloc[1].fillna("")
    etf_names = {}
    for idx in range(1, etf_raw.shape[1], 2):
        ticker = str(ticker_row.iloc[idx]).strip()
        if not ticker:
            continue
        label = str(name_row.iloc[idx])
        etf_names[ticker] = _normalize_etf_label(label)

    return RawData(
        risk_free=risk_free,
        hedge_fund=hedge_fund,
        hedge_fund_names=hedge_fund_names,
        etf_raw=etf_raw,
        etf_names=etf_names,
    )


def _prepare_risk_free(
    raw_rf: pd.DataFrame, processing: ProcessingConfig
) -> pd.DataFrame:
    rf = raw_rf.copy()
    rf["Date"] = pd.to_datetime(rf["Date"], format="%Y%m%d")
    rf = rf.set_index("Date").sort_index()
    rf = rf.resample(processing.pandas_frequency).sum()
    rf = (rf / 100.0).apply(np.log1p)
    return rf.loc[processing.start_ts : processing.end_ts]


def _prepare_hedge_funds(
    raw_hfd: pd.DataFrame, rf: pd.DataFrame, processing: ProcessingConfig
) -> pd.DataFrame:
    data = raw_hfd.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")
    data = data.drop_duplicates(subset="Date", keep="last")
    data = data.set_index("Date")
    if not data.empty:
        data = data.iloc[1:]
    for column in data.columns:
        cleaned = data[column].astype(str).str.rstrip("%")
        data[column] = pd.to_numeric(cleaned, errors="coerce") / 100.0
    data = data.loc[processing.start_ts : processing.end_ts]
    data = data.dropna(how="any")
    log_returns = data.apply(np.log1p)
    combined = log_returns.join(rf, how="inner")
    rf_series = combined.pop("RF")
    excess = combined.subtract(rf_series, axis=0)
    return excess


def _prepare_factor_etf(
    raw_etf: pd.DataFrame,
    etf_names: Dict[str, str],
    rf: pd.DataFrame,
    processing: ProcessingConfig,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    data_rows = raw_etf.iloc[2:].reset_index(drop=True)
    ticker_row = raw_etf.iloc[1].fillna("")
    series_dict: Dict[str, pd.Series] = {}
    for idx in range(0, raw_etf.shape[1], 2):
        value_idx = idx + 1
        if value_idx >= raw_etf.shape[1]:
            break
        ticker = str(ticker_row.iloc[value_idx]).strip()
        if not ticker or ticker not in etf_names:
            continue
        if ticker in processing.drop_etf_symbols:
            continue
        subset = data_rows.iloc[:, [idx, value_idx]].copy()
        subset.columns = ["Date", ticker]
        subset["Date"] = _parse_dates(subset["Date"])
        subset[ticker] = pd.to_numeric(subset[ticker], errors="coerce")
        subset = subset.dropna()
        if subset.empty:
            continue
        series = subset.set_index("Date")[ticker].sort_index()
        series = series[~series.index.duplicated(keep="last")]
        relative = series / series.shift(1)
        log_ret = relative.map(np.log).dropna()
        monthly = (
            log_ret.resample(processing.pandas_frequency)
            .sum()
            .loc[processing.start_ts : processing.end_ts]
        )
        if monthly.empty:
            continue
        series_dict[ticker] = monthly

    if not series_dict:
        raise ValueError("No ETF series produced; check drop list and raw data")

    factor_etf: pd.DataFrame = pd.DataFrame(series_dict)
    factor_etf = factor_etf.sort_index()
    factor_etf = factor_etf.join(rf, how="inner")
    rf_series = factor_etf.pop("RF")
    factor_etf = factor_etf.subtract(rf_series, axis=0)
    factor_etf = factor_etf.dropna(how="any")
    filtered_names = {key: etf_names[key] for key in factor_etf.columns}
    return factor_etf, filtered_names


def clean(raw: RawData, config: DataPrepConfig) -> PreparedData:
    """Transform raw inputs into aligned monthly log excess returns."""

    rf = _prepare_risk_free(raw.risk_free, config.processing)
    hedge_funds = _prepare_hedge_funds(raw.hedge_fund, rf, config.processing)
    factor_etf, factor_names = _prepare_factor_etf(
        raw.etf_raw, raw.etf_names, rf, config.processing
    )
    return PreparedData(
        risk_free=rf,
        hedge_funds=hedge_funds,
        factor_etf=factor_etf,
        hedge_fund_names={k: raw.hedge_fund_names[k] for k in hedge_funds.columns},
        factor_etf_names=factor_names,
    )


def align_monthly(data: PreparedData) -> PreparedData:
    """Intersect indices to keep datasets perfectly aligned."""

    index = data.risk_free.index
    for frame in (data.hedge_funds, data.factor_etf):
        index = index.intersection(frame.index)
    aligned_rf = data.risk_free.loc[index]
    aligned_hf = data.hedge_funds.loc[index]
    aligned_factor = data.factor_etf.loc[index]
    return PreparedData(
        risk_free=aligned_rf,
        hedge_funds=aligned_hf,
        factor_etf=aligned_factor,
        hedge_fund_names=data.hedge_fund_names,
        factor_etf_names=data.factor_etf_names,
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_to_save = frame.copy()
    frame_to_save.index.name = "Date"
    frame_to_save.to_csv(path)


def _write_pickle(path: Path, payload: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def save_prepared(data: PreparedData, config: DataPrepConfig) -> None:
    """Persist prepared datasets and mirror them to legacy directories."""

    datasets = config.outputs.dataset_paths()
    metadata = config.outputs.metadata_paths()

    logger.info("Writing hedge fund data to %s", datasets["hedge_funds"])
    _write_frame(datasets["hedge_funds"], data.hedge_funds)
    logger.info("Writing factor ETF data to %s", datasets["factor_etf"])
    _write_frame(datasets["factor_etf"], data.factor_etf)
    logger.info("Writing risk-free series to %s", datasets["risk_free"])
    _write_frame(datasets["risk_free"], data.risk_free)

    logger.info("Serializing hedge fund names to %s", metadata["hedge_fund_names"])
    _write_pickle(metadata["hedge_fund_names"], data.hedge_fund_names)
    logger.info("Serializing factor ETF names to %s", metadata["factor_etf_names"])
    _write_pickle(metadata["factor_etf_names"], data.factor_etf_names)

    for mirror in config.outputs.mirror_dirs():
        mirror.mkdir(parents=True, exist_ok=True)
        for key, path in datasets.items():
            target = mirror / path.name
            logger.info("Mirroring %s to %s", key, target)
            _write_frame(target, pd.read_csv(path, index_col=0, parse_dates=True))
        for key, path in metadata.items():
            target = mirror / path.name
            logger.info("Mirroring metadata %s to %s", key, target)
            with path.open("rb") as source, target.open("wb") as sink:
                sink.write(source.read())


def prepare(config_path: Path | str = "configs/data.yaml") -> PreparedData:
    """High-level helper: load, transform, align, and persist datasets."""

    config = DataPrepConfig.from_yaml(config_path)
    raw = load_raw(config)
    cleaned = clean(raw, config)
    aligned = align_monthly(cleaned)
    save_prepared(aligned, config)
    return aligned


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    prepare()


__all__ = [
    "DataPrepConfig",
    "PreparedData",
    "RawData",
    "load_raw",
    "clean",
    "align_monthly",
    "save_prepared",
    "prepare",
]


if __name__ == "__main__":
    main()
