# -*- coding: utf-8 -*-
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_engine import FaultAnalyzer  # noqa: E402
from sample_data import build_sample_dataframe  # noqa: E402

# 日付を固定してテストを再現可能にする
BASE_DATE = "2026-07-01"


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return build_sample_dataframe(base_date=BASE_DATE)


@pytest.fixture(scope="session")
def analyzer(sample_df) -> FaultAnalyzer:
    return FaultAnalyzer(sample_df.copy())
