import numpy as np
from omnixai.data.image import Image
from omnixai.preprocessing.image import Resize


class ScoreCAMMixin:

    @staticmethod
    def _resize_scores(inputs, scores, channel_last=False):
        if not channel_last:
            size = inputs.shape[2:] if inputs.shape[1] == 3 else inputs.shape[1:3]
        else:
            size = inputs.shape[1:3]
        for i in range(scores.shape[0]):
            min_val, max_val = np.min(scores[i]), np.max(scores[i])
            scores[i] = (scores[i] - min_val) / (max_val - min_val + 1e-8) * 255
        im = Resize(size).transform(Image(data=scores, batched=True))
        return im.to_numpy() / 255.0

    @staticmethod
    def _resize_image(image, inputs):
        assert image.shape[0] == 1, "`image` can contain one instance only."
        y = image.to_numpy()
        x = inputs
        if not isinstance(x, np.ndarray):
            try:
                x = x.detach().cpu().numpy()
            except:
                x = x.numpy()
        x = x.squeeze()
        if x.shape[0] == 3:
            x = np.transpose(x, (1, 2, 0))

        min_a, max_a = np.min(y), np.max(y)
        min_b, max_b = np.min(x), np.max(x)
        r = (max_a - min_a) / (max_b - min_b + 1e-8)
        return Image(data=(r * x + min_a - r * min_b).astype(int), batched=False, channel_last=True)
