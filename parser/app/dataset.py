"""Чтение официального XLSX заказчика и преобразование в SignalDoc."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from shared.schema import SignalDoc, Source, SourceType, TrustLevel

ROOT = Path(os.getenv("DATASET_DIR", "/app/data"))
DEFAULT_NAMES = ("official_dataset.xlsx", "weak_signals.xlsx", "dataset.xlsx")


def find_dataset() -> Optional[Path]:
    explicit = os.getenv("OFFICIAL_DATASET_PATH")
    if explicit and Path(explicit).exists():
        return Path(explicit)
    for name in DEFAULT_NAMES:
        p = ROOT / name
        if p.exists():
            return p
    return None


def _md_links(text: str):
    return re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", text or "")


def _stage(value: str) -> Optional[int]:
    text = (value or "").lower()
    if "ранн" in text or "внедрен" in text:
        return 4
    if "пилот" in text:
        return 3
    if "прототип" in text or "poc" in text:
        return 2
    if "концеп" in text or "исслед" in text:
        return 1
    return None


def _trend(value: str) -> Optional[int]:
    text = (value or "").lower()
    if "быстро" in text:
        return 3
    if "раст" in text:
        return 2
    return 1 if text else None


def load_docs(limit: int = 100) -> List[SignalDoc]:
    path = find_dataset()
    if not path:
        return []
    import pandas as pd

    df = pd.read_excel(path, sheet_name="Слабые сигналы", header=1)
    column_aliases = {
        "Технология": ["Технология", "Технология (слабый сигнал)"],
        "Область": ["Область"],
        "Компании": ["Компании"],
        "Почему это слабый сигнал": ["Почему это слабый сигнал"],
        "Стадия развития": ["Стадия развития"],
        "Тренд упоминаний": ["Тренд упоминаний"],
        "Балл (стадия+тренд)": ["Балл (стадия+тренд)"],
        "Источники": ["Источники"],
    }
    resolved = {}
    missing = []
    for canonical, aliases in column_aliases.items():
        match = next((name for name in aliases if name in df.columns), None)
        if match is None:
            missing.append(canonical)
        else:
            resolved[canonical] = match
    if missing:
        raise ValueError("В XLSX отсутствуют колонки: {}".format(", ".join(missing)))

    docs = []
    for idx, row in df.head(limit).iterrows():
        companies = [x.strip() for x in re.split(r"[,;]\s*", str(row[resolved["Компании"]])) if x.strip()]
        sources = []
        for title, url in _md_links(str(row[resolved["Источники"]])):
            sources.append(Source(
                title=title or "Источник из датасета",
                url=url,
                date=None,
                source_type=SourceType.SCIENTIFIC,
                language="ru",
                trust_level=TrustLevel.MEDIUM,
                translated=False,
            ))
        docs.append(SignalDoc(
            id="dataset-{:03d}".format(idx + 1),
            title=str(row[resolved["Технология"]]),
            area=str(row[resolved["Область"]]),
            companies=companies,
            raw_text=str(row[resolved["Почему это слабый сигнал"]]),
            stage=_stage(str(row[resolved["Стадия развития"]])),
            trend=_trend(str(row[resolved["Тренд упоминаний"]])),
            dataset_score=int(row[resolved["Балл (стадия+тренд)"]]) if str(row[resolved["Балл (стадия+тренд)"]]).strip() else None,
            why=str(row[resolved["Почему это слабый сигнал"]]),
            is_weak_signal=True,
            sources=sources,
        ))
    return docs
