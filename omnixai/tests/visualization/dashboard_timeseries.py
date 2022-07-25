import os
import unittest
import numpy as np
import pandas as pd
from omnixai.data.timeseries import Timeseries
from omnixai.explainers.timeseries import TimeseriesExplainer
from omnixai.visualization.dashboard import Dashboard


def load_timeseries():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets")
    df = pd.read_csv(os.path.join(data_dir, "timeseries.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
    df = df.rename(columns={"horizontal": "values"})
    df = df.set_index("timestamp")
    df = df.drop(columns=["anomaly"])
    return df


def train_detector(train_df):
    threshold = np.percentile(train_df["values"].values, 90)

    def _detector(ts: Timeseries):
        anomaly_scores = np.sum((ts.values > threshold).astype(int))
        return anomaly_scores / ts.shape[0]

    return _detector


class TestDashboard(unittest.TestCase):

    def setUp(self) -> None:
        df = load_timeseries()
        self.train_df = df.iloc[:9150]
        self.test_df = df.iloc[9150:9300]
        self.detector = train_detector(self.train_df)
        print(self.detector(Timeseries.from_pd(self.test_df)))

    def test(self):
        explainers = TimeseriesExplainer(
            explainers=["shap", "mace"],
            mode="anomaly_detection",
            data=Timeseries.from_pd(self.train_df),
            model=self.detector,
            preprocess=None,
            postprocess=None,
            params={"mace": {"threshold": 0.001}}
        )
        test_instance = Timeseries.from_pd(self.test_df)
        local_explanations = explainers.explain(test_instance)
        print(local_explanations)
        dashboard = Dashboard(instances=test_instance, local_explanations=local_explanations)
        dashboard.show()


if __name__ == "__main__":
    unittest.main()
