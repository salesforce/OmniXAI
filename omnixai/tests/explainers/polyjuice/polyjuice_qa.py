import unittest
from transformers import pipeline

from omnixai.data.text import Text
from omnixai.explainers.nlp.counterfactual.polyjuice import Polyjuice


class TestPolyjuice(unittest.TestCase):

    def setUp(self) -> None:
        model_name = "deepset/roberta-base-squad2"
        self.model = pipeline('question-answering', model=model_name, tokenizer=model_name)

        def _predict(x: Text):
            x = x.split(sep="[SEP]")
            inputs = [{"context": y[0], "question": y[1]} for y in x]
            outputs = self.model(inputs)
            if isinstance(outputs, dict):
                outputs = [outputs]
            return [output["answer"] for output in outputs]

        self.explainer = Polyjuice(predict_function=_predict, mode="qa")

    def test_explain(self):
        x = Text([
            "The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks. [SEP] "
            "What can people do with model coversion?",
            "Electric vehicles emit much less harmful pollutants than conventional vehicles and ultimately, create a cleaner environment for human beings. [SEP] "
            "what is the result of using eletric vehicles?"
        ])
        explanations = self.explainer.explain(x)

        from omnixai.visualization.dashboard import Dashboard
        dashboard = Dashboard(
            instances=x,
            local_explanations={"polyjuice": explanations}
        )
        dashboard.show()


if __name__ == "__main__":
    unittest.main()
