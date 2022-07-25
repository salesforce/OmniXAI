import unittest
import pandas as pd
from datetime import datetime
from typing import List
from merlion.utils.time_series import TimeSeries as merlion_timeSeries
from omnixai.data.timeseries import Timeseries as omnixai_timeSeries
from merlion.models.defaults import DefaultForecasterConfig, DefaultForecaster
from omnixai.explainers.timeseries.agnostic.shap import ShapTimeseries


class TestMerlion(unittest.TestCase):

    def setUp(self) -> None:
        # Load the time-series data
        url = "https://raw.githubusercontent.com/AmirMK/XAI_Time_Seris_SF/main/my_test_data.csv"
        df = pd.read_csv(url)
        time = df["Month of Datetime"].tolist()
        target = df["#Passengers"].tolist()

        # Run data pre-processing
        time = [datetime.strptime(date[:10], "%Y-%m-%d").date() for date in time]
        df = pd.DataFrame({"Datetime": time, "Target": target})
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")
        df.index = pd.to_datetime(df.index)
        df.index.freq = "MS"
        df = df[~df["Target"].isna()]

        # Train data with merlion_timeSeries as merlion is used to build the model
        self.train_data = merlion_timeSeries.from_pd(df.iloc[:-10])
        self.test_data = df.iloc[-10:]
        self.test_data["Target"] = 1

        # Create and train the model
        self.model = DefaultForecaster(DefaultForecasterConfig())
        self.model.train(self.train_data)

    def test(self):

        def forecasting_function(ts: omnixai_timeSeries):
            # If time_stamps has only one timeseries, "to_pd" returns pd.DataFrame.
            # The output is the forecasted value.
            df = ts.to_pd()
            targets, _ = self.model.forecast(
                time_stamps=1,
                time_series_prev=merlion_timeSeries.from_pd(df)
            )
            return list(targets.items())[0][1].values[0]

        # Create shap explnaier
        explainers = ShapTimeseries(
            training_data=omnixai_timeSeries.from_pd(self.train_data.to_pd()),
            predict_function=forecasting_function,
            mode="forecasting"
        )
        test_data = omnixai_timeSeries.from_pd(self.test_data.iloc[:5])
        explanations = explainers.explain(test_data, nsamples=100)
        print(explanations)


if __name__ == "__main__":
    unittest.main()
