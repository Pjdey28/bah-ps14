from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pyarrow_parquet
except Exception:  # pragma: no cover - optional dependency
    pyarrow_parquet = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_ROOT = Path(os.getenv("SOLAR_MODELS_ROOT", PROJECT_ROOT / "artifacts" / "models"))
DATA_ROOT = Path(os.getenv("SOLAR_DATA_ROOT", PROJECT_ROOT / "data"))
PROCESSED_DATA = DATA_ROOT / "processed"
SEQUENCE_LENGTH = int(os.getenv("SOLAR_SEQUENCE_LENGTH", "72"))
STREAM_LIMIT_ROWS = int(os.getenv("SOLAR_STREAM_LIMIT_ROWS", "384"))
TARGETS = ["target_30min", "target_6hr", "target_12hr"]


class SolarStormReplaySource:
    def __init__(self, limit_rows: int | None = None):
        self.limit_rows = limit_rows or STREAM_LIMIT_ROWS
        self.mode = "historical"
        self.pointer = 0
        self.predictor = self._load_predictor()
        self.frame = self._load_frame()
        self.timeline = self._build_timeline()

    def _load_predictor(self):
        if os.getenv("SOLAR_USE_MODELS", "1").lower() in {"0", "false", "no"}:
            return None

        try:
            import src.config as repo_config

            repo_config.MODELS = MODELS_ROOT
            from src.inference.predictor import SolarStormPredictor

            if MODELS_ROOT.exists():
                self.mode = "model"
                return SolarStormPredictor()
        except Exception:
            return None

        return None

    def _read_parquet_head(self, file_path: Path, rows: int) -> pd.DataFrame:
        if pyarrow_parquet is not None:
            parquet_file = pyarrow_parquet.ParquetFile(str(file_path))
            chunks = []
            remaining = rows

            for row_group_index in range(parquet_file.num_row_groups):
                if remaining <= 0:
                    break

                chunk = parquet_file.read_row_group(row_group_index).to_pandas()
                if len(chunk) > remaining:
                    chunk = chunk.head(remaining)

                chunks.append(chunk)
                remaining -= len(chunk)

            if chunks:
                return pd.concat(chunks, ignore_index=True)

        frame = pd.read_parquet(file_path, engine="pyarrow")
        return frame.head(rows)

    def _load_frame(self) -> pd.DataFrame:
        train_dir = PROCESSED_DATA / "train"
        files = sorted(train_dir.glob("train_*.parquet"))

        if not files:
            raise FileNotFoundError(f"No parquet files found in {train_dir}.")

        per_file = max(1, math.ceil(self.limit_rows / max(1, len(files))))
        frames = []

        for file_path in files:
            try:
                frame = self._read_parquet_head(file_path, per_file)
            except Exception:
                continue

            if not frame.empty:
                frames.append(frame)

            if sum(len(item) for item in frames) >= self.limit_rows:
                break

        if not frames:
            raise RuntimeError("Unable to read any training parquet rows.")

        frame = pd.concat(frames, ignore_index=True)
        if "time" in frame.columns:
            frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
            frame = frame.sort_values("time")

        frame = frame.drop_duplicates().reset_index(drop=True)
        return frame.head(self.limit_rows)

    def _predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.predictor is None or len(frame) <= SEQUENCE_LENGTH:
            replay = frame.iloc[SEQUENCE_LENGTH:].copy().reset_index(drop=True)
            for target_name, source_name in zip(
                ["forecast_30min", "forecast_6hr", "forecast_12hr"],
                TARGETS,
            ):
                if source_name in replay.columns:
                    replay[target_name] = np.expm1(pd.to_numeric(replay[source_name], errors="coerce"))
                else:
                    replay[target_name] = np.nan
            return replay

        try:
            predicted = self.predictor.predict(frame)
        except Exception:
            self.predictor = None
            return self._predict_frame(frame)

        predicted = predicted.copy()
        predicted["forecast_30min"] = np.expm1(pd.to_numeric(predicted["predicted_30min"], errors="coerce"))
        predicted["forecast_6hr"] = np.expm1(pd.to_numeric(predicted["predicted_6hr"], errors="coerce"))
        predicted["forecast_12hr"] = np.expm1(pd.to_numeric(predicted["predicted_12hr"], errors="coerce"))
        return predicted[["time", "forecast_30min", "forecast_6hr", "forecast_12hr"]]

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        low = values.min()
        high = values.max()

        if pd.isna(low) or pd.isna(high) or high <= low:
            return pd.Series(np.zeros(len(values)), index=series.index)

        return (values - low) / (high - low)

    def _build_timeline(self) -> pd.DataFrame:
        forecast_frame = self._predict_frame(self.frame)
        timeline = self.frame.iloc[SEQUENCE_LENGTH:].copy().reset_index(drop=True)

        if len(forecast_frame) != len(timeline):
            size = min(len(forecast_frame), len(timeline))
            timeline = timeline.iloc[:size].copy().reset_index(drop=True)
            forecast_frame = forecast_frame.iloc[:size].copy().reset_index(drop=True)

        if timeline.empty:
            return pd.DataFrame()

        timeline = pd.concat(
            [timeline, forecast_frame.drop(columns=["time"], errors="ignore")],
            axis=1,
        )

        timeline["risk"] = self._normalize(timeline["forecast_12hr"])
        timeline["alert_level"] = np.where(
            timeline["risk"] >= 0.72,
            "ALERT",
            np.where(timeline["risk"] >= 0.48, "WATCH", "NORMAL"),
        )
        timeline["sequence"] = np.arange(1, len(timeline) + 1)
        return timeline.reset_index(drop=True)

    @staticmethod
    def _pick(row: pd.Series, candidates: list[str], default=0):
        for candidate in candidates:
            value = row.get(candidate)
            if pd.notna(value):
                return value
        return default

    def _row_to_packet(self, row: pd.Series) -> dict:
        risk = float(row.get("risk", 0.0))
        return {
            "timestamp": str(row.get("time", "")),
            "goes": "GOES/WIND",
            "wind": "WIND",
            "risk": round(risk, 2),
            "speed": round(float(self._pick(row, ["flow_speed", "flow_speed_mean1h", "flow_speed_lag6"], 0)), 1),
            "density": round(float(self._pick(row, ["proton_density", "proton_density_mean1h", "proton_density_lag6"], 0)), 1),
            "imf": round(float(self._pick(row, ["B_total", "BZ_GSE", "VBz"], 0)), 1),
            "kp": int(min(9, max(1, round(risk * 9) or 1))),
            "alert": str(row.get("alert_level", "WATCH")),
            "sequence": int(row.get("sequence", 0)),
            "forecast_30min": round(float(row.get("forecast_30min", 0.0)), 3),
            "forecast_6hr": round(float(row.get("forecast_6hr", 0.0)), 3),
            "forecast_12hr": round(float(row.get("forecast_12hr", 0.0)), 3),
            "mode": self.mode,
        }

    def next_packet(self) -> dict:
        if self.timeline.empty:
            return {
                "timestamp": "",
                "goes": "GOES/WIND",
                "wind": "WIND",
                "risk": 0.0,
                "speed": 0.0,
                "density": 0.0,
                "imf": 0.0,
                "kp": 1,
                "alert": "NORMAL",
                "sequence": 0,
                "forecast_30min": 0.0,
                "forecast_6hr": 0.0,
                "forecast_12hr": 0.0,
                "mode": self.mode,
            }

        row = self.timeline.iloc[self.pointer]
        packet = self._row_to_packet(row)
        self.pointer = (self.pointer + 1) % len(self.timeline)
        return packet

    def snapshot(self, window: int = 48) -> dict:
        if self.timeline.empty:
            return {"packet": self.next_packet(), "series": [], "alerts": [], "mode": self.mode, "rows": 0}

        series = [
            {
                "label": row.get("time").strftime("%Y-%m-%d %H:%M") if pd.notna(row.get("time")) else f"Pkt {idx}",
                "risk": float(row.get("risk", 0.0)),
            }
            for idx, (_, row) in enumerate(self.timeline.tail(window).iterrows(), start=1)
        ]

        alerts = self.timeline[self.timeline["risk"] >= 0.48].sort_values("risk", ascending=False).head(6)
        alert_records = [
            {
                "id": int(row.get("sequence", idx)),
                "title": "High solar storm alert" if row.get("risk", 0.0) >= 0.72 else "Geomagnetic watch",
                "message": f"Forecast risk is {float(row.get('risk', 0.0)):.2f}.",
                "level": str(row.get("alert_level", "WATCH")).lower(),
                "time": str(row.get("time", "")),
            }
            for idx, (_, row) in enumerate(alerts.iterrows(), start=1)
        ]

        return {
            "packet": self._row_to_packet(self.timeline.iloc[(self.pointer - 1) % len(self.timeline)]),
            "series": series,
            "alerts": alert_records,
            "mode": self.mode,
            "rows": int(len(self.timeline)),
        }