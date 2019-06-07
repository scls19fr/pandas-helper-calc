#!/usr/bin/env python3

import pandas as pd


@pd.api.extensions.register_series_accessor("calc")
class CalcAccessor(object):
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        pass

    def derivative(self):
        index = self._obj.index
        if isinstance(index, pd.DatetimeIndex):
            den = index.to_series(keep_tz=True).diff().dt.total_seconds()
        else:
            den = index.to_series().diff()
        num = self._obj.diff()
        return num.div(den, axis=0)
    
    def integrate(self, start=None, end=None, var=None):
        index = self._obj.index

        if var is None:
            if isinstance(index, pd.DatetimeIndex):
                s_index = pd.Series(index)
                s = (((self._obj + self._obj.shift()) / 2.0) * s_index.diff().dt.total_seconds().values).fillna(0).cumsum()
            else:
                s_index = pd.Series(index)
                s = (((self._obj + self._obj.shift()) / 2.0) * s_index.diff().values).fillna(0).cumsum()
        else:
            raise NotImplementedError("Can't integrate with variable different from index")

        if start is None and end is None:
            return s
        elif start is not None and end is not None:
            raise NotImplementedError("init and stop can both be set")
        else:
            if start is not None:
                return s + start
            elif end is not None:
                return s - s.iloc[-1] + end


@pd.api.extensions.register_dataframe_accessor("calc")
class CalcAccessor(object):
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        pass

    def derivative(self, var=None):
        if var is None:
            index = self._obj.index
            if isinstance(index, pd.DatetimeIndex):
                den = self._obj.index.to_series(keep_tz=True).diff().dt.total_seconds()
            else:
                den = index.to_series().diff()
            num = self._obj.diff()
            return num.div(den, axis=0)
        else:
            if pd.api.types.is_datetime64_any_dtype(self._obj[var]):
                den = self._obj[var].diff().dt.total_seconds()
            else:
                den = self._obj[var].diff()
            num = self._obj.loc[:, self._obj.columns != var].diff()
            result = num.div(den, axis=0)
            result[var] = self._obj[var]
            return result.loc[:, self._obj.columns]

    def integrate(self, start=None, end=None):
        newobj = self._obj.copy()
        for column in newobj.columns:
            newobj[column] = newobj[column].calc.integrate(start, end)
        return newobj