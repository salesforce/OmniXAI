#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import unittest
import pandas as pd
from omnixai.data.timeseries import Timeseries


class TestTimeseries(unittest.TestCase):

    def setUp(self) -> None:
        df = pd.DataFrame(
            [['2017-12-27', 1263.94091, 394.507, 16.530, 411.037],
             ['2017-12-28', 1299.86398, 506.424, 14.162, 520.586]],
            columns=['Date', 'Consumption', 'Wind', 'Solar', 'Wind+Solar']
        )
        df = df.set_index('Date')
        df.index = pd.to_datetime(df.index)
        self.df = df

    def test(self):
        ts = Timeseries.from_pd(self.df)
        self.assertEqual(ts.ts_len, 2)
        self.assertEqual(ts.shape, (2, 4))
        self.assertEqual(len(ts), 2)
        df = ts.to_pd()
        self.assertEqual(df.index.year.values[0], 2017)
        self.assertEqual(df.index.month.values[0], 12)
        x = ts.to_numpy()
        self.assertEqual(x[1][1], 506.424)
        x = ts[[1, 0]]
        self.assertEqual(x.values[1][1], 394.507)

        timestamp_info = Timeseries.get_timestamp_info(ts)
        ts_a = Timeseries.reset_timestamp_index(ts, timestamp_info)
        ts_b = Timeseries.restore_timestamp_index(ts_a, timestamp_info)
        self.assertListEqual(list(ts.columns), list(ts_b.columns))
        self.assertListEqual(list(ts.index), list(ts_b.index))
        self.assertEqual(ts.to_pd().equals(ts_b.to_pd()), True)


if __name__ == "__main__":
    unittest.main()
