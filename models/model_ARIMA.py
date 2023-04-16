import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ARIMA_model:
    def __init__(self) -> None:
        pass
    def predict(self, sequence, p=1, d=0, q=1):
        # 将用户行为序列转换为Pandas DataFrame
        data = pd.Series(sequence, dtype=float)
        model = ARIMA(data, order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1).reset_index(drop=True)
        return int(round(forecast[0]))
