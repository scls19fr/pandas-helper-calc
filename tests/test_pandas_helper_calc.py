import pytest
import pandas_helper_calc

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal


# Derivative


def test_derivative_Series_with_float_index():
    x = np.arange(0, 2 * np.pi + 0.05, 0.05)
    y1 = pd.Series(np.sin(x), index=x)
    y2 = pd.Series(np.cos(x), index=x)

    dy1dx = y1.calc.derivative()

    assert ((dy1dx - y2).fillna(0).abs() > 0.05).sum() == 0


def test_derivative_DataFrame_with_float_index():
    x = np.arange(0, 2 * np.pi + 0.05, 0.05)
    y1 = np.sin(x)
    y2 = np.cos(x)

    df = pd.DataFrame({"y1": y1, "y2": y2}, index=x)
    df.index.name = "x"

    dy1dx = df["y1"].calc.derivative()

    assert ((dy1dx - df["y2"]).fillna(0).abs() > 0.05).sum() == 0

    # test derivative using column name
    dfdt = df.reset_index().calc.derivative(var="x")
    assert dfdt.iloc[-1]["y1"] == pytest.approx(1.0, 0.001)
    # assert dfdt.iloc[-1]['y2'] == pytest.approx(0.0, 0.1)


def test_derivative_Series_with_DateTimeIndex():
    t = np.arange(0, 2 * np.pi + 0.05, 0.05)
    y1 = pd.Series(np.sin(t), index=t)
    t = pd.to_datetime("2019-01-01") + pd.to_timedelta(t, unit="s")
    dy1dt = y1.calc.derivative()


def test_derivative_DataFrame_with_DateTimeIndex():
    t = np.arange(0, 2 * np.pi + 0.05, 0.05)
    y1 = np.sin(t)
    y2 = np.cos(t)
    t = pd.to_datetime("2019-01-01") + pd.to_timedelta(t, unit="s")
    df = pd.DataFrame({"y1": y1, "y2": y2}, index=t)
    df.index.name = "t"

    dfdt = df.calc.derivative()

    # assert_series_equal(dfdt.index, df.index)


# Integrate


def test_integrate_Series_with_float_index():
    x = np.arange(0, 2 * np.pi + 0.05, 0.05)
    y1 = pd.Series(np.sin(x), index=x)
    y2 = pd.Series(-np.cos(x), index=x)

    Int_y1 = y1.calc.integrate(-1)

    assert_series_equal(Int_y1, y2, check_less_precise=1)


def test_integrate_DataFrame_with_float_index():
    x = np.arange(0, 2 * np.pi + 0.05, 0.05)
    y1 = np.sin(x)
    y2 = np.cos(x)

    df = pd.DataFrame({"y1": y1, "y2": y2}, index=x)

    df.calc.integrate(-1)


def test_integrate_Series_with_DateTimeIndex():
    x = np.arange(0, 2 * np.pi + 0.05, 0.05)
    t = pd.to_datetime("2019-01-01") + pd.to_timedelta(x, unit="s")

    y1 = pd.Series(np.sin(x), index=t)
    y2 = pd.Series(-np.cos(x), index=t)

    y1.calc.integrate(-1)


def test_integrate_DataFrame_with_DateTimeIndex():
    x = np.arange(0, 2 * np.pi + 0.05, 0.05)
    t = pd.to_datetime("2019-01-01") + pd.to_timedelta(x, unit="s")

    y1 = np.sin(x)
    y2 = np.cos(x)

    df = pd.DataFrame({"y1": y1, "y2": y2}, index=x)

    df.calc.integrate(-1)
