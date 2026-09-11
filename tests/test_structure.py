"""Tests for structure detection service and rules."""

import pandas as pd
import numpy as np
import pytest

from models.structure import StructureType
from services.structure.service import StructureService
from services.structure.detector import StructureDetector


def test_detect_normal_table() -> None:
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "customer_name": ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Hannah", "Ian", "Jack"],
        "age": [25, 30, 45, 22, 38, 52, 29, 33, 41, 27],
        "signup_date": pd.to_datetime(["2026-01-01"] * 10),
        "is_active": [True, False, True, True, False, True, True, False, True, True],
        "account_balance": [120.5, 450.0, 310.2, 0.0, 95.5, 1200.0, 34.0, 560.8, 890.1, 412.0],
    })

    result = StructureService.detect(df)
    assert result.structure in [StructureType.NORMAL_TABLE, StructureType.TRANSACTIONAL]
    assert result.confidence > 0.4


def test_detect_wide_table() -> None:
    # Wide dataset with categories as columns (e.g. Sales by region)
    df = pd.DataFrame({
        "region_id": ["R1", "R2", "R3", "R4", "R5"],
        "sales_jan": [100, 150, 200, 120, 180],
        "sales_feb": [110, 160, 210, 125, 190],
        "sales_mar": [105, 155, 205, 130, 185],
        "sales_apr": [115, 165, 215, 135, 195],
        "sales_may": [120, 170, 220, 140, 200],
        "sales_jun": [125, 175, 225, 145, 205],
        "sales_jul": [130, 180, 230, 150, 210],
        "sales_aug": [135, 185, 235, 155, 215],
    })

    result = StructureService.detect(df)
    assert result.structure == StructureType.WIDE_TABLE
    assert result.confidence > 0.4


def test_detect_pivot_table() -> None:
    df = pd.DataFrame({
        "Category": ["Electronics", "Clothing", "Home", "Garden"],
        "North": [1000, 500, 750, 300],
        "South": [1200, 600, 800, 350],
        "East": [900, 450, 700, 280],
        "West": [1100, 550, 780, 320],
    })

    result = StructureService.detect(df)
    assert result.structure in [StructureType.PIVOT_TABLE, StructureType.WIDE_TABLE]
    assert result.confidence > 0.4


def test_detect_multi_header() -> None:
    # Multi-header where row 0 contains labels / headers
    df = pd.DataFrame({
        "Unnamed: 0": ["Store Name", "Store A", "Store B", "Store C"],
        "Unnamed: 1": ["Quarter 1", 100, 200, 300],
        "Unnamed: 2": ["Quarter 2", 150, 250, 350],
        "Unnamed: 3": ["Quarter 3", 120, 220, 320],
    })

    result = StructureService.detect(df)
    assert result.structure == StructureType.MULTI_HEADER
    assert result.confidence > 0.4


def test_detect_time_series() -> None:
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=30, freq="D"),
        "temperature": np.random.randn(30) + 25,
        "humidity": np.random.randn(30) + 60,
    })

    result = StructureService.detect(df)
    assert result.structure == StructureType.TIME_SERIES
    assert result.confidence > 0.4
