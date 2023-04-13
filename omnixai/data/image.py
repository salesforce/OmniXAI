#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The class for image data.
"""
import numpy as np
from PIL import Image as PilImage
from typing import Union, List

from .base import Data


class Image(Data):
    """
    The class represents a batch of images. It supports both grayscale and RGB images.
    It will convert the input images into the `(batch_size, h, w, channel)` format. If
    there is only one input image, `batch_size` will be 1.
    """

    data_type = "image"

    def __init__(
            self, data: Union[np.ndarray, PilImage.Image] = None, batched: bool = False, channel_last: bool = True
    ):
        """
        :param data: The image data, which is either np.ndarray or PIL.Image. If ``data``
            is a numpy array, it should have the following format: `(h, w, channel)`, `(channel, h, w)`,
            `(batch_size, h, w, channel)` or `(batch_size, channel, h, w)`.
            If ``data`` is a PIL.Image, ``batched`` and ``channel_last`` are ignored.
            The images contained in ``data`` will be automatically converted into a numpy array with
            shape `(batch_size, h, w, channel)`. If there is only one image, `batch_size` will be 1.
        :param batched: `True` if the first dimension of ``data`` is the batch size.
            `False` if ``data`` has one image only.
        :param channel_last: `True` if the last dimension of ``data`` is the color channel
            or `False` if the first or second dimension of ``data`` is the color channel. If ``data``
            has no color channel, e.g., grayscale images, this argument is ignored.
        """
        super().__init__()
        if data is None:
            self.data = None
        elif isinstance(data, np.ndarray):
            self.data = self._check_and_unify(data, batched, channel_last)
        elif isinstance(data, PilImage.Image):
            self.data = self._check_and_unify(np.array(data), batched=False, channel_last=True)
        else:
            raise ValueError(f"`data` should have type `np.ndarray` or `PIL.Image` " f"instead of {type(data)}")

    @staticmethod
    def _check_and_unify(data: np.ndarray, batched: bool, channel_last: bool):
        """
        Checks the data format and converts the data into the shape `(batch_size, h, w, channel)`.
        If the input has one image only, the shape will be (1, h, w, channel).

        :param data: The raw data of the image.
        :param batched: `True` if the first dimension of ``data`` is the batch size.
            `False` if ``data`` has one image only.
        :param channel_last: `True` if the last dimension is the color channel
            or `False` if the first dimension is the color channel. If ``data`` has no color channel,
            e.g., grayscale images, this argument is ignored.
        """
        if batched:
            assert data.ndim == 3 or data.ndim == 4, (
                f"Because `batched = {batched}`, the dimension of an image "
                f"should have shape (batch_size, *, *) or (batch_size, *, *, *) "
                f"instead of {data.shape}"
            )
            img = data[0]
        else:
            assert data.ndim == 2 or data.ndim == 3, (
                f"Because `batched = {batched}`, the dimension of an image "
                f"should have shape (*, *) or (*, *, *) "
                f"instead of {data.shape}"
            )
            img = data

        if img.ndim == 3:
            if channel_last:
                assert img.shape[2] <= 4, (
                    f"`channel_last = {channel_last}`, the last dimension of " f"`data` should be the color channels."
                )
            else:
                assert img.shape[0] <= 4, (
                    f"`channel_last = {channel_last}`, the first dimension of " f"`data` should be the color channels."
                )

        if not batched:
            data = np.expand_dims(data, axis=0)
        if data.ndim == 4:
            if not channel_last:
                data = np.transpose(data, (0, 2, 3, 1))
        else:
            data = np.expand_dims(data, axis=-1)
        return data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self):
        return repr(self.data)

    def __getitem__(self, i: Union[int, slice, list]):
        """
        Gets a subset of images given the indices.

        :param i: An integer index or slice.
        :return: A subset of images.
        :rtype: Image
        """
        if isinstance(i, int):
            return Image(self.data[i: i + 1], batched=True, channel_last=True)
        else:
            return Image(self.data[i], batched=True, channel_last=True)

    def __iter__(self):
        return (self.__getitem__(i) for i in range(self.shape[0]))

    @property
    def shape(self) -> tuple:
        """
        Returns the raw data shape.

        :return: A tuple for the raw data shape, e.g., `(batch_size, h, w, channel)`.
        :rtype: tuple
        """
        return self.data.shape

    def num_samples(self) -> int:
        """
        Returns the number of the images.

        :return: The number of the images.
        :rtype: int
        """
        return self.data.shape[0]

    @property
    def image_shape(self) -> tuple:
        """
        Returns the image shape.

        :return: A tuple for the image shape, e.g., `(h, w, channel)`.
        :rtype: tuple
        """
        return self.data.shape[1:]

    @property
    def values(self) -> np.ndarray:
        """
        Returns the raw values.

        :return: A numpy array of the stored images.
        :rtype: np.ndarray
        """
        return self.data

    def to_numpy(self, hwc=True, copy=True, keepdim=False) -> np.ndarray:
        """
        Converts `Image` into a numpy ndarray.

        :param hwc: The output has format `(batch_size, h, w, channel)` if `hwc` is True
            or `(batch_size, channel, h, w)` otherwise.
        :param copy: `True` if it returns a data copy, or `False` otherwise.
        :param keepdim: `True` if the number of dimensions is kept for grayscale images,
            `False` if the channel dimension is squeezed.
        :return: A numpy ndarray representing the images.
        :rtype: np.ndarray
        """
        data = self.data.copy() if copy else self.data
        if not keepdim:
            if self.shape[-1] > 1:
                return data if hwc else np.transpose(data, (0, 3, 1, 2))
            else:
                return data.squeeze(axis=-1)
        else:
            return data if hwc else np.transpose(data, (0, 3, 1, 2))

    def to_pil(self) -> Union[PilImage.Image, List]:
        """
        Converts `Image` into a Pillow image or a list of Pillow images.

        :return: A single Pillow image if `batch_size = 1` or a list of Pillow images
            if `batch_size > 1`.
        :rtype: Union[PilImage.Image, List]
        """
        x = self.data.squeeze(axis=-1) if self.shape[-1] == 1 else self.data
        if self.shape[0] == 1:
            return PilImage.fromarray(x[0].astype(np.uint8))
        else:
            return [PilImage.fromarray(x[i].astype(np.uint8)) for i in range(self.shape[0])]

    def copy(self):
        """
        Returns a copy of the image data.

        :return: The copied image data.
        :rtype: Image
        """
        return Image(data=self.data.copy(), batched=True, channel_last=True)
