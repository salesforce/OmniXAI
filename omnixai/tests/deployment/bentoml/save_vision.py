#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import torch
from torchvision import models, transforms

from omnixai.explainers.vision import VisionExplainer
from omnixai.deployment.bentoml.omnixai import save_model, load_model


def test_save_and_load():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def _preprocess(ims):
        import torch
        return torch.stack([transform(im.to_pil()) for im in ims]).to(device)

    def _postprocess(logits):
        import torch
        return torch.nn.functional.softmax(logits, dim=1)

    model = models.resnet50(pretrained=True).to(device)

    explainer = VisionExplainer(
        explainers=["gradcam", "layercam", "smoothgrad"],
        mode="classification",
        model=model,
        preprocess=_preprocess,
        postprocess=_postprocess,
        params={
            "gradcam": {"target_layer": model.layer4[-1]},
            "layercam": {"target_layer": model.layer3[-3]},
        },
    )
    save_model("vision_explainer", explainer)
    print("Save explainer successfully.")
    explainer = load_model("vision_explainer:latest")
    print(explainer)
    print("Load explainer successfully.")


if __name__ == "__main__":
    test_save_and_load()
