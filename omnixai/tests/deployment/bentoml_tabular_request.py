import unittest
import requests


class TestSklearnRequest(unittest.TestCase):

    def test(self):
        result = requests.post(
            "http://0.0.0.0:3000/predict",
            headers={"content-type": "application/json"},
            data='[[5,4,3,2]]'
        ).text
        print(result)


if __name__ == "__main__":
    unittest.main()
