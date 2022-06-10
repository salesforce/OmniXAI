omnixai.data package
======================

.. automodule:: omnixai.data
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   base
   tabular
   image
   text
   timeseries

This package provides classes for representing tabular data, image data, text and time series data,
i.e., :py:mod:`omnixai.data.tabular`, :py:mod:`omnixai.data.image`, :py:mod:`omnixai.data.text` and
:py:mod:`omnixai.data.timeseries`, respectively.

Given a pandas dataframe ``df`` with a set of categorical column names ``categorical_columns`` and
a target column ``target_column`` (e.g., class labels), we can create a ``Tabular`` object as follows:

.. code-block:: python

   from omnixai.data.tabular import Tabular

   tabular = Tabular(
       data=df,                                  # a pandas dataframe
       categorical_columns=categorical_columns,  # a list of categorical feature names
       target_column=target_column)              # a target column name

If ``df`` has no categorical columns or no target column, we can set ``categorical_columns=None`` or
``target_column=None``, respectively. We can also create a ``Tabular`` object with a numpy array ``x``
with a list of feature names ``feature_columns`` and categorical feature names ``categorical_columns``:

.. code-block:: python

   from omnixai.data.tabular import Tabular

   tabular = Tabular(
       data=x,                                   # a numpy array
       feature_columns=feature_columns,          # a list of feature names
       categorical_columns=categorical_columns)  # a list of categorical feature names

If there are no feature names, the default feature names will be the indices in the numpy array, e.g., 0, 1, ...

The ``Image`` class represents a batch of images. Given a batch of images stored in a numpy array ``x`` with shape
`(batch_size, height, width, channel)`, we can create an ``Image`` instance:

.. code-block:: python

   from omnixai.data.image import Image

   images = Image(
       data=x,             # a numpy array with shape (batch_size, height, width, channel)
       batched=True,       # if x represents a batch of images
       channel_last=True)  # if the last dimension of x is `channel`

If the last dimension is not `channel`, namely, ``x`` has shape `(batch_size, channel, height, width)`, we need
to set ``channel_last=False`` instead. If the numpy array ``x`` has only one image with shape `(height, width, channel)`,
we need to set ``batched=False`` because the number of dimensions in ``x`` is 3 instead of 4.

We can also convert a ``Pillow`` image into an ``Image`` instance:

.. code-block:: python

   from PIL import Image as PilImage
   from omnixai.data.image import Image

   im = PilImage.open("an_image.jpg")
   image = Image(data=im)

The ``Text`` class represents a batch of texts or sentences. Given a list of strings ``texts``, we can create
an ``Text`` instance:

.. code-block:: python

   from omnixai.data.text import Text
   text = Text(data=["Hello I'm a single sentence",
                     "And another sentence",
                     "And the very very last one"])

The ``Text`` class also allows to specify the tokenizer to split each text/sentence into tokens via the ``tokenizer``
parameter. If ``tokenizer`` is set to `None`, a default tokenizer ``nltk.word_tokenize`` is applied.

The ``Timeseries`` class represents a batch of time series. The values of metrics/variables are stored in a numpy array
with shape `(batch_size, timestamps, num_variables)`. If there is only one time series, `batch_size` is 1.
We can construct a ``Timeseries`` instance from one or a list of pandas dataframes. The index of the dataframe
indicates the timestamps and the columns are the variables.

.. code-block:: python

   from omnixai.data.timeseries import Timeseries
   df = pd.DataFrame(
       [['2017-12-27', 1263.94091, 394.507, 16.530],
        ['2017-12-28', 1299.86398, 506.424, 14.162]],
       columns=['Date', 'Consumption', 'Wind', 'Solar']
   )
   df = df.set_index('Date')
   df.index = pd.to_datetime(df.index)
   ts = Timeseries.from_pd(self.df)

omnixai.data.base module
--------------------------

.. automodule:: omnixai.data.base
   :members:
   :undoc-members:
   :show-inheritance:

omnixai.data.tabular module
---------------------------

.. automodule:: omnixai.data.tabular
   :members:
   :undoc-members:
   :show-inheritance:

omnixai.data.image module
--------------------------

.. automodule:: omnixai.data.image
   :members:
   :undoc-members:
   :show-inheritance:

omnixai.data.text module
--------------------------

.. automodule:: omnixai.data.text
   :members:
   :undoc-members:
   :show-inheritance:

omnixai.data.timeseries module
------------------------------

.. automodule:: omnixai.data.timeseries
   :members:
   :undoc-members:
   :show-inheritance:
