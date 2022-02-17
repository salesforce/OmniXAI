omnixai.preprocessing package
=============================

.. automodule:: omnixai.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   base
   encode
   fill
   normalize
   pipeline
   tabular
   image
   text

This package provides a number of useful data pre-processing transforms. Each transform
inherits from :py:mod:`omnixai.preprocessing.base.TransformBase` with three main methods:

- **fit(self, x)**: Estimates the parameters of the transform with data ``x``.
- **transform(self, x)**: Applies the transform to the input data ``x``.
- **invert(self, x)**: Applies the inverse transform to the input data ``x``.

For example, :py:mod:`omnixai.preprocessing.tabular.TabularTransform` provides a convenient way for feature
pre-processing on tabular datasets:

.. code-block:: python

    from omnixai.data.tabular import Tabular
    from omnixai.preprocessing.normalize import MinMax
    from omnixai.preprocessing.encode import OneHot
    from omnixai.preprocessing.tabular import TabularTransform

    x = Tabular(
        data=pd.DataFrame({
            'A': [1, 2, 2, 6],
            'B': [5, 4, 3, 2],
            'C': ['a', 'b', 'c', 'd']
        }),
        categorical_columns=['C']
    )
    transform = TabularTransform(
        cate_transform=OneHot(),        # One-hot encoding for categorical features
        cont_transform=MinMax()         # Min-max normalization for continuous-valued features
    ).fit(x)
    y = transform.transform(x)          # Transforms tabular data into a numpy array
    z = transform.invert(y)             # Applies the inverse transform

Note that some transforms such as `FillNaN`, `FillNaNTabular` only have *pseudo*-inverse transforms
that may not recover the original data.

For `Image` data, one can transform images in a similar way:

.. code-block:: python

    from PIL import Image as PilImage
    from omnixai.data.image import Image
    from omnixai.preprocessing.image import Resize

    img = Image(PilImage.open('some_image.jpg'))
    transform = Resize(size=(360, 240))             # A transform for resizing images
    x = transformer.transform(img)                  # Applies the transform
    y = transformer.invert(x)                       # Applies the inverse transform

For `Text` data, one can apply a TF-IDF transform as follows:

.. code-block:: python

    from omnixai.data.text import Text
    from omnixai.preprocessing.text import Tfidf

    text = Text(
        data=["Hello I'm a single sentence",
              "And another sentence",
              "And the very very last one"]
    )
    transform = Tfidf().fit(text)                   # Fit a TF-IDF transform
    vectors = transform.transform(text)             # Applies the transform for feature vectors

omnixai.preprocessing.base module
-------------------------------------

.. automodule:: omnixai.preprocessing.base
   :members:
   :undoc-members:
   :show-inheritance:

omnixai.preprocessing.encode module
-------------------------------------

.. automodule:: omnixai.preprocessing.encode
   :members:
   :undoc-members:
   :show-inheritance:

omnixai.preprocessing.normalize module
--------------------------------------

.. automodule:: omnixai.preprocessing.normalize
   :members:
   :undoc-members:
   :show-inheritance:

omnixai.preprocessing.fill module
-------------------------------------

.. automodule:: omnixai.preprocessing.fill
   :members:
   :undoc-members:
   :show-inheritance:

omnixai.preprocessing.pipeline module
-------------------------------------

.. automodule:: omnixai.preprocessing.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

omnixai.preprocessing.tabular module
-------------------------------------

.. automodule:: omnixai.preprocessing.tabular
   :members:
   :undoc-members:
   :show-inheritance:

omnixai.preprocessing.image module
-------------------------------------

.. automodule:: omnixai.preprocessing.image
   :members:
   :undoc-members:
   :show-inheritance:

omnixai.preprocessing.text module
-------------------------------------

.. automodule:: omnixai.preprocessing.text
   :members:
   :undoc-members:
   :show-inheritance:

omnixai.sampler.tabular module
-------------------------------------

.. automodule:: omnixai.sampler.tabular
   :members:
   :undoc-members:
   :show-inheritance:
