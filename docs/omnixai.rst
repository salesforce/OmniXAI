OmniXAI: An Explanation Toolbox
===============================

Overview
--------

:py:mod:`omnixai` is a Python library for explainable AI (XAI). The library provides
a comprehensive family of explainable AI capabilities and interpretable machine learning algorithms,
including high-quality implementations of various model-agnostic and model-specific explanation methods,
e.g., feature-importance explanation, counterfactual explanation, gradient-based explanation, which supports
tabular data, image data and text data. It has five key subpackages:

- :py:mod:`omnixai.data`: This package contains the classes for representing tabular, image, text, and time series data,
  i.e., :py:mod:`omnixai.data.tabular`, :py:mod:`omnixai.data.image`, :py:mod:`omnixai.data.text` and :py:mod:`omnixai.data.timeseries`,
  respectively. For example, the explainers for tabular data use an instance of :py:mod:`omnixai.data.tabular` as one of their inputs.
  The library provides simple constructors for creating instances of these classes from numpy arrays, pandas dataframes,
  pillow images or strings.
- :py:mod:`omnixai.preprocessing`: This package contains various pre-processing functions for different feature types:

  - One-hot encoding and ordinal encoding for categorical features.
  - KBins, standard normalization, min-max normalization, rescaling, NaN-filling for continuous-valued features.
  - A pre-processing module :py:mod:`omnixai.preprocessing.tabular` for tabular data.
  - Recaling, normalization, resizing for image data.
  - The TF-IDF transformation and token-to-id transformation for text data.
  - A pre-processing pipeline :py:mod:`omnixai.preprocessing.pipeline` combining multiple pre-processing functions together.

  For tabular data, :py:mod:`omnixai.preprocessing.tabular.TabularTransform` provides a convenient way for feature pre-processing.
  One can simply use this class to transform the raw data into the training/test dataset that a particular machine learning
  model can consume.

- :py:mod:`omnixai.explainers`: This is the main package in the library, which contains all the supported explainers.
  The explainers are categorized into four groups:

  - :py:mod:`omnixai.explainers.data`: It is for data exploration/analysis, including feature correlation analysis,
    feature imbalance analysis, feature selection, etc.
  - :py:mod:`omnixai.explainers.tabular`: It contains the explainers for tabular data, e.g., global explanations
    such as PDP, local explanations such as LIME, SHAP, MACE.
  - :py:mod:`omnixai.explainers.vision`: It contains the explainers for vision tasks, e.g., integrated-gradient,
    Grad-CAM, contrastive explanation, counterfactual explanation.
  - :py:mod:`omnixai.explainers.nlp`: It contains the explainers for NLP tasks, e.g., LIME, integrated-gradient.
  - :py:mod:`omnixai.explainers.timeseries`: It contains the explainers for time series tasks, e.g., SHAP, MACE.

For each group, the explainers are further categorized into "model-agnostic", "model-specific" and "counterfactual".
A "model-agnostic" explainer can handle black-box ML models, i.e., only requiring a prediction function without
knowing model details. A "model-specific" explainer requires some information of ML models, e.g., whether the model is
differentiable, whether the model is a linear model or a tree-based model. "counterfactual" is a special group for counterfactual
explanation methods which may be either "model-agnostic" or "model-specific".

- :py:mod:`omnixai.explanations`: This package contains the classes for explanation results. For example,
  :py:mod:`omnixai.explanations.tabular.feature_importance` is used for storing feature-importance/attribution explanations.
  All of these classes provide plotting functions for visualization, e.g., "plot" using "Matplotlib", "plotly_plot" using "Dash"
  and "ipython_plot" for IPython. Explanations are categorized into three groups:

  - :py:mod:`omnixai.explanations.tabular`: For tabular explainers, e.g., feature-importance explanation, etc.
  - :py:mod:`omnixai.explanations.image`: For vision explainers, e.g., pixel-importance explanation, etc.
  - :py:mod:`omnixai.explanations.text`: For NLP explainers, e.g., word/token-importance explanation, etc.
  - :py:mod:`omnixai.explanations.timeseries`: For time series explainers, e.g., counterfactual explanation, etc.

- :py:mod:`omnixai.visualization`: This package provides a dashboard for visualization implemented using Plotly Dash. The
  dashboard supports both global explanations and local explanations.

Installation
------------

You can install :py:mod:`omnixai` from PyPI by calling ``pip install omnixai``. You may install from source by
cloning the OmniXAI repo, navigating to the root directory, and calling
``pip install .``, or ``pip install -e .`` to install in editable mode. You may install additional dependencies:

- **For vision tasks**: Calling ``pip install omnixai[vision]``, or ``pip install .[vision]`` from the
  root directory of the repo.
- **For NLP tasks**: Calling ``pip install omnixai[nlp]``, or ``pip install .[nlp]`` from the
  root directory of the repo.
- **For plotting & visualization**: Calling ``pip install omnixai[plot]``, or ``pip install .[plot]`` from the
  root directory of the repo.

Getting Started
---------------

To get started, we recommend the linked tutorials in :ref:`Tutorials & Example Code <tutorial>`.
In general, we recommend using :py:mod:`omnixai.explainers.tabular.TabularExplainer`, :py:mod:`omnixai.explainers.vision.VisionExplainer`,
:py:mod:`omnixai.explainers.nlp.NLPExplainer` and :py:mod:`omnixai.explainers.timeseries.TimeseriesExplainer` for tabular, vision, NLP and
time series tasks, respectively. To generate explanations, one only needs to specify

- **The ML model to explain**: e.g., a scikit-learn model, a tensorflow model, a pytorch model or a black-box prediction function.
- **The pre-processing function**: i.e., converting raw data into the model inputs.
- **The post-processing function (optional)**: i.e., converting the model outputs into class probabilities.
- **The explainers to apply**: e.g., SHAP, MACE, Grad-CAM.

Let's take the income prediction task as an example.
The dataset used in this example is for income prediction (https://archive.ics.uci.edu/ml/datasets/adult).
We recommend using data class `Tabular` to represent a tabular dataset. To create a `Tabular` instance given a pandas
dataframe, one needs to specify the dataframe, the categorical feature names (if exists) and the target/label
column name (if exists).

.. code-block:: python

   from omnixai.data.tabular import Tabular
   # Load the dataset
   feature_names = [
       "Age", "Workclass", "fnlwgt", "Education",
       "Education-Num", "Marital Status", "Occupation",
       "Relationship", "Race", "Sex", "Capital Gain",
       "Capital Loss", "Hours per week", "Country", "label"
   ]
   df = pd.DataFrame(
      np.genfromtxt('adult.data', delimiter=', ', dtype=str),
      columns=feature_names
   )
   tabular_data = Tabular(
       df,
       categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
       target_column='label'
   )

The package :py:mod:`omnixai.preprocessing` provides several useful preprocessing functions
for a `Tabular` instance. `TabularTransform` is a special transform designed for processing tabular data.
By default, it converts categorical features into one-hot encoding, and keeps continuous-valued features.
The  method ``transform`` of `TabularTransform` transforms a `Tabular` instance to a numpy array.
If the `Tabular` instance has a target/label column, the last column of the numpy array
will be the target/label. After data preprocessing, we can train a XGBoost classifier for this task.

.. code-block:: python

   from omnixai.preprocessing.tabular import TabularTransform
   # Data preprocessing
   transformer = TabularTransform().fit(tabular_data)
   class_names = transformer.class_names
   x = transformer.transform(tabular_data)
   # Train an XGBoost model (the last column of `x` is the label column after transformation)
   model = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
   model.fit(x[:, :-1], x[:, -1])

To initialize `TabularExplainer`, we need to set the following parameters:

- ``explainers``: The names of the explainers to apply, e.g., ["lime", "shap", "mace", "pdp"].
- ``data``: The data used to initialize explainers. ``data`` is the training dataset for training the
  machine learning model. If the training dataset is too large, ``data`` can be a subset of it by applying
  :py:mod:`omnixai.sampler.tabular.Sampler.subsample`.
- ``model``: The ML model to explain, e.g., a scikit-learn model, a tensorflow model or a pytorch model.
- ``preprocess``: The preprocessing function converting the raw data into the inputs of ``model``.
- ``postprocess`` (optional): The postprocessing function transforming the outputs of ``model`` to a
  user-specific form, e.g., the predicted probability for each class.
- ``mode``: The task type, e.g., "classification" or "regression".

The preprocessing function takes a `Tabular` instance as its input and outputs the processed features that
the ML model consumes. In this example, we simply call ``transformer.transform``.

.. code-block:: python

   from omnixai.explainers.tabular import TabularExplainer
   from omnixai.visualization.dashboard import Dashboard

   # Initialize a TabularExplainer
   explainers = TabularExplainer(
      explainers=["lime", "shap", "mace", "pdp"],       # The explainers to apply
      mode="classification",                            # The task type
      data=tabular_data,                                # The data for initializing the explainers
      model=model,                                      # The ML model to explain
      preprocess=lambda z: transformer.transform(z),    # Converts raw features into the model inputs
      params={
         "mace": {"ignored_features": ["Sex", "Race", "Relationship", "Capital Loss"]}
      }                                                 # Additional parameters
   )

In this example, LIME, SHAP and MACE generate local explanations while PDP (partial dependence plot)
generates global explanations. ``explainers.explain`` returns the local explanations generated by the
three methods given the test instances, and ``explainers.explain_global`` returns the global explanations
generated by PDP. `TabularExplainer` hides all the details behind the explainers, so we can simply call
these two methods to generate explanations.

.. code-block:: python

   # Generate explanations
   test_instances = tabular_data[:5]
   local_explanations = explainers.explain(X=test_instances)
   global_explanations = explainers.explain_global()

Given the generated explanations, we can launch a dashboard (a Dash app) for visualization by setting the test
instance, the generated local explanations, the generated global explanations, the class names, and additional
parameters for visualization (e.g., only plotting the selected features in PDP).

.. code-block:: python

   # Launch a dashboard for visualization
   dashboard = Dashboard(
       instances=test_instances,                        # The instances to explain
       local_explanations=local_explanations,           # Set the generated local explanations
       global_explanations=global_explanations,         # Set the generated global explanations
       class_names=class_names,                         # Set class names
       params={"pdp": {"features": ["Age", "Education-Num", "Capital Gain",
                                    "Capital Loss", "Hours per week", "Education",
                                    "Marital Status", "Occupation"]}}
   )
   dashboard.show()                                     # Launch the dashboard


After opening the Dash app in the browser, we will see a dashboard showing the explanations:

.. image:: _static/demo.png

How to Contribute
-----------------

Thank you for your interest in contributing to the library!
Before you get started, clone this repo, run ``pip install pre-commit``, and run ``pre-commit install`` from the root
directory of the repo. This will ensure all files are formatted correctly and contain the appropriate
license headers whenever you make a commit. To add a new explanation method into the library,
one may follow the steps below:

#. Choose the task type of the new explainer, e.g., "tabular", "vision", "nlp" or "timeseries".
#. Choose the explainer type, e.g., "model-agnostic", "model-specific" or "counterfactual".
#. Create a new python script file for this explainer in the specified folder, e.g., it is put
   under the folder "explainers/tabular/agnostic" if it is a model-agnostic explainer for tabular data.
#. Create the explainer class that inherits from :py:mod:`omnixai.explainers.base.ExplainerBase`.
   The constructor for the new explainer class has two options:

   - **__init__(self, predict_function, mode, **kwargs)**: This is for model-agnostic explainers. ``predict_function``
     is the prediction function of the black-box ML model to explain. The inputs of ``predict_function`` are the raw
     input features, and the outputs of ``predict_function`` are the model outputs. ``mode`` is the task type, e.g.,
     "classification", "regression".
   - **__init__(self, model, preprocess_function, postprocess_function, mode, **kwargs)**: This is for model-specific explainers.
     ``model`` is the ML model to explain. The model-specific explainers require
     some information about ``model``, e.g., whether ``model`` is differentiable (PyTorch or Tensorflow). ``preprocess_function``
     is the pre-processing function for ``model``, converting the raw features into the inputs of ``model``, e.g., resizing
     images to (224, 224) and normalizing pixel values. ``postprocess_function`` is the post-processing function for ``model``,
     which is used to convert the output logits into class probabilities. ``postprocess_function`` is optional.
     ``mode`` is the task type, e.g., "classification", "regression".

#. Add a class attribute ``explanation_type`` (string) with value "local", "global" or "both", indicating whether the method
   can generate local explanations, global explanations or both.
#. Add a class attribute ``alias`` (list) specifying the explainer names.
#. Implement the "explain" function, e.g., ``explain(self, **kwargs)`` for local explanations, or
   ``explain_global(self, X, **kwargs)`` for global explanations where the type of ``X`` is class `Tabular`, `Image`, `Text` or `Timeseries`.
#. Import the explainer class in "__init__.py" of the packages :py:mod:`omnixai.explainers.tabular`,
   :py:mod:`omnixai.explainers.vision`, :py:mod:`omnixai.explainers.nlp` or :py:mod:`omnixai.explainers.timeseries`.

The new explainer will be registered automatically, which can be called via :py:mod:`omnixai.explainers.tabular.TabularExplainer`,
:py:mod:`omnixai.explainers.vision.VisionExplainer`, :py:mod:`omnixai.explainers.nlp.NLPExplainer` or :py:mod:`omnixai.explainers.timeseries.TimeseriesExplainer`
by specifying one of the names defined in ``alias``.

.. automodule:: omnixai
   :members:
   :undoc-members:
   :show-inheritance:

Modules for Different Data Types
--------------------------------

.. toctree::
   :maxdepth: 4

   omnixai.data

Preprocessing Functions
-----------------------

.. toctree::
   :maxdepth: 4

   omnixai.preprocessing

Supported Explanation Methods
-----------------------------

.. toctree::
   :maxdepth: 4

   omnixai.explainers

Modules for Explanation Results
-------------------------------

.. toctree::
   :maxdepth: 4

   omnixai.explanations

Dashboard for Visualization
---------------------------

.. toctree::
   :maxdepth: 4

   omnixai.visualization
