import os
import torch
import unittest
from PIL import Image as PilImage
from omnixai.data.text import Text
from omnixai.data.image import Image
from omnixai.data.multi_inputs import MultiInputs
from omnixai.preprocessing.image import Resize
from omnixai.explainers.vision_language.specific.ig import IntegratedGradient

from lavis.models import BlipITM
from lavis.processors import load_processor


class TestIG(unittest.TestCase):

    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")
        image = Resize(size=480).transform(
            Image(PilImage.open(os.path.join(directory, "images/girl_dog.jpg")).convert("RGB")))
        text = Text(["A girl playing with her dog on the beach"])
        self.inputs = MultiInputs(image=image, text=text)

        pretrained_path = \
            "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth"
        self.model = BlipITM(pretrained=pretrained_path, vit="base")
        self.image_processor = load_processor("blip_image_eval").build(image_size=384)
        self.text_processor = load_processor("blip_caption")
        self.tokenizer = BlipITM.init_tokenizer()

        def _preprocess(x: MultiInputs):
            images = torch.stack([self.image_processor(z.to_pil()) for z in x.image])
            texts = [self.text_processor(z) for z in x.text.values]
            return images, texts

        self.preprocess = _preprocess

    def test(self):
        explainer = IntegratedGradient(
            model=self.model,
            embedding_layer=self.model.text_encoder.embeddings.word_embeddings,
            preprocess_function=self.preprocess,
            tokenizer=self.tokenizer,
            loss_function=lambda outputs: outputs[:, 1].sum()
        )
        explanations = explainer.explain(self.inputs)
        fig = explanations.plotly_plot()
        fig.show()


if __name__ == "__main__":
    unittest.main()
