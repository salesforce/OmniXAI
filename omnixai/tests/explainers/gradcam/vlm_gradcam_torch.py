import os
import torch
import unittest
import numpy as np
from PIL import Image as PilImage
from omnixai.data.text import Text
from omnixai.data.image import Image
from omnixai.data.multi_inputs import MultiInputs
from omnixai.preprocessing.image import Resize
from omnixai.explainers.vision_language.specific.gradcam import GradCAM

from lavis.models import BlipITM
from lavis.processors import load_processor


class TestGradCAM(unittest.TestCase):

    def setUp(self) -> None:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../datasets")
        im = Resize(size=480).transform(
            Image(PilImage.open(os.path.join(directory, "images/girl_dog.jpg")).convert("RGB")))
        image = Image(data=np.concatenate([im.to_numpy(), im.to_numpy()]), batched=True)
        text = Text(["A girl playing with her dog on the beach", "A girl playing with her dog"])
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
        explainer = GradCAM(
            model=self.model,
            target_layer=self.model.text_encoder.base_model.base_model.encoder.layer[6].
                crossattention.self.attention_probs_layer,
            preprocess_function=self.preprocess,
            tokenizer=self.tokenizer,
            loss_function=lambda outputs: outputs[:, 1].sum()
        )
        explanations = explainer.explain(self.inputs)
        fig = explanations.plotly_plot()
        fig.show()


if __name__ == "__main__":
    unittest.main()
