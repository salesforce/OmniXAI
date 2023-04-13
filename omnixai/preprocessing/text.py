#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The pre-processing functions for text data.
"""
import sklearn.feature_extraction

from .base import TransformBase
from ..data.text import Text


class Tfidf(TransformBase):
    """
    The TF-IDF transformation.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.tfidf = sklearn.feature_extraction.text.TfidfVectorizer(**kwargs)

    def fit(self, x: Text, **kwargs):
        self.tfidf.fit(x.values)
        return self

    def transform(self, x: Text):
        assert self.tfidf is not None, "The TFIDF model is not trained."
        return self.tfidf.transform(x.values)

    def invert(self, x):
        raise RuntimeError("The TFIDF transformer doesn't support `invert`.")

    def get_feature_names(self):
        """
        Returns the feature names in the transformed data.
        """
        return self.tfidf.get_feature_names()


class Word2Id(TransformBase):
    """
    The class for converting words into IDs.
    """

    PAD = 0
    START = 1
    UNK = 2

    def __init__(self, remove_punctuation=True, **kwargs):
        super().__init__()
        self.remove_punctuation = remove_punctuation
        self.word_to_id = {}
        self.id_to_word = {}

    def fit(self, x: Text, **kwargs):
        self.word_to_id = {"<PAD>": self.PAD, "<START>": self.START, "<UNK>": self.UNK}
        texts = x.to_tokens(**kwargs)
        counts = {}
        for tokens in texts:
            if self.remove_punctuation:
                tokens = [w for w in tokens if w.isalnum()]
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
        counts = sorted(counts.items(), key=lambda c: c[1], reverse=True)
        for i, (token, _) in enumerate(counts):
            self.word_to_id[token] = i + 3
        self.id_to_word = {val: key for key, val in self.word_to_id.items()}
        return self

    def transform(self, x: Text, **kwargs):
        assert len(self.word_to_id) > 0, "The vocabulary is empty, please run `fit` first."
        texts = x.to_tokens(**kwargs)
        if self.remove_punctuation:
            token_ids = [[self.word_to_id.get(token, self.UNK) for token in text if token.isalnum()] for text in texts]
        else:
            token_ids = [[self.word_to_id.get(token, self.UNK) for token in text] for text in texts]
        return token_ids

    def invert(self, x):
        assert len(self.id_to_word) > 0, "The vocabulary is empty, please run `fit` first."
        tokens = [[self.id_to_word.get(i, self.id_to_word[self.UNK]) for i in ids] for ids in x]
        return tokens

    @property
    def vocab_size(self):
        return len(self.word_to_id)
