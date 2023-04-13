#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The class for text data.
"""
from typing import List, Union, Callable
from .base import Data
from ..utils.misc import is_nltk_available


if is_nltk_available():
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
else:
    nltk = None


class Text(Data):
    """
    The class represents a batch of texts or sentences.
    The texts or sentences are stored in a list of strings.
    """

    data_type = "text"

    def __init__(self, data: Union[List, str] = None, tokenizer: Callable = None):
        """
        :param data: The text data, either a string or a list of strings.
        :param tokenizer: A tokenizer for splitting texts/sentences into tokens,
            which should be `Callable` object. If `tokenizer` is None, a default
            `nltk` tokenizer will be applied.
        """
        super().__init__()
        if data is None:
            self.data = []
        elif isinstance(data, str):
            self.data = [data]
        else:
            self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, i: Union[int, slice]):
        """
        Gets a subset of texts given the indices.

        :param i: An integer index or slice.
        :return: A subset of texts.
        :rtype: Text
        """
        return Text(self.data[i], tokenizer=self.tokenizer)

    def __iter__(self):
        return (self.__getitem__(i) for i in range(len(self.data)))

    def num_samples(self) -> int:
        """
        Returns the number of the texts or sentences.

        :return: The number of the texts or sentences.
        :rtype: int
        """
        return len(self.data)

    @property
    def values(self):
        """
        Returns the raw text data.

        :return: A list of the sentences/texts.
        :rtype: List
        """
        return self.data

    def to_tokens(self, **kwargs) -> List:
        """
        Converts sentences/texts into tokens.
        If `tokenizer` is None, a default `split` function, e.g., `nltk.word_tokenize` is called
        to split a sentence into tokens. For example, `["omnixai library", "explainable AI"]` will
        be split into `[["omnixai", "library"], ["explainable", "AI"]]`.

        :param kwargs: Additional parameters for the tokenizer
        :return: A batch of tokens.
        :rtype: List
        """
        if self.tokenizer is None:
            if nltk is None:
                raise ImportError("Package `nltk` is not installed.")
            return [nltk.word_tokenize(s.lower()) for s in self.data]
        else:
            return self.tokenizer(self.data, **kwargs)

    def to_str(self, copy=True) -> Union[List, str]:
        """
        Returns a string if it has only one sentence or
        a list of strings if it contains multiple sentences.

        :param copy: Whether to copy the data.
        :return: A single string or a list of strings.
        :rtype: Union[List, str]
        """
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data.copy() if copy else self.data

    def split(self, sep=None, maxsplit=-1):
        return [s.split(sep, maxsplit) for s in self.data]

    def copy(self):
        """
        Returns a copy of the text data.

        :return: The copied text data.
        :rtype: Text
        """
        return Text(data=self.data.copy(), tokenizer=self.tokenizer)
