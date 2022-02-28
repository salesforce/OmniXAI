.. OmniXAI documentation master file, created by
   sphinx-quickstart on Wed Oct  6 13:42:55 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OmniXAI's documentation!
========================================

Introduction
############

OmniXAI is a Python library for explainable AI and interpretable machine learning.
The library provides a comprehensive family of explainable AI capabilities and interpretable
machine learning algorithms. It includes high-quality implementations of various model-agnostic and
model-specific explanation methods, e.g., feature-attribution explanation, counterfactual explanation,
gradient-based explanation, and supports tabular data, image data and text data. It also provides an
easy-to-use interface that allows users to obtain explanations by writing few lines of codes,
which can be applied for multiple purposes in various applications:

#. **Data exploration**: What are highly correlated features? Does there exist data imbalance issues?
#. **Feature engineering**: Which features are important for the task? How to improve the model performance?
#. **Model debugging**: Does the model utilize wrong features? Why does the model make a mistake?
#. **Decision support for business applications**: How to understand the model’s decision and why to trust them?
#. **Transparency for healthcare and finance**: Why does the model make such judgement? Is the judgement reasonable?

Compared with other existing explanation libraries (such as IBM’s AIX360, Microsoft’s InterpretML, Alibi and explainX),
our library has a comprehensive list of XAI capabilities and unique features include the followings:

#. **Data analysis/exploration**: Analyzing feature correlations, checking imbalance issues.
#. **Support most popular explanation methods**: Many explanation methods can be applied to analyze different aspects of a ML model.
#. **Support counterfactual explanation**: Providing the information about how to change the current prediction.
#. **Support gradient-based explanation**: Integrated-gradient, Grad-CAM and Grad-CAM++ are included.
#. **Support Image and text data**: Our library provides various explanations for image/text models.
#. **A much simpler interface**: Users only need to write few lines of code to generate various explanations.
#. **A GUI dashboard**: Providing an GUI dashboard for users to examine the generated explanations.
#. **Easy to extend**: Developers can add new explanation algorithms easily by implementing a single class derived from
   the explainer base class.

Capabilities and Features
#########################

The following table shows the supported explanation methods and features in our library.
We will continue improving this library to make it more comprehensive in the future, e.g., supporting more
explanation methods for vision and NLP tasks.

=======================  ====================  ================  =============  =======  =======  =======
Method                   Model Type            Explanation Type  Data Analysis  Tabular  Image    Text
=======================  ====================  ================  =============  =======  =======  =======
Feature correlation      NA                    Global            ✓
Feature imbalance        NA                    Global            ✓
Partial dependence       Black box             Global                           ✓
Sensitivity analysis     Black box             Global                           ✓
LIME                     Black box             Local                            ✓        ✓        ✓
SHAP                     Black box*            Local                            ✓        ✓        ✓
Integrated gradient      Torch or TF           Local                            ✓        ✓        ✓
Counterfactual           Black box*            Local                            ✓        ✓
Contrastive explanation  Torch or TF           Local                                     ✓
Grad-CAM, Grad-CAM++     Torch or TF           Local                                     ✓
Learning to explain      Black box             Local                            ✓        ✓        ✓
Linear models            Linear models         Global and Local                 ✓
Tree models              Tree models           Global and Local                 ✓
=======================  ====================  ================  =============  =======  =======  =======

*SHAP* accepts black box models for tabular data, PyTorch/Tensorflow models for image data, transformer models
for text data. *Counterfactual* accepts black box models for tabular data and PyTorch/Tensorflow models for
image data.

Comparison with Competitors
###########################

The following table shows the comparison between our toolkit/library and other existing XAI toolkits/libraries
in literature:

=========  ====================  =======  ===========  ======  ====  ======  =====  ========
Data Type  Method                OmniXAI  InterpretML  AIX360  Eli5  Captum  Alibi  explainX
=========  ====================  =======  ===========  ======  ====  ======  =====  ========
Tabular    LIME                  ✓        ✓            ✓       ✘     ✓       ✘      ✘
\          SHAP                  ✓        ✓            ✓       ✘     ✓       ✓      ✓
\          Partial dependence    ✓        ✓            ✘       ✘     ✘       ✘      ✘
\          Sensitivity           ✓        ✓            ✘       ✘     ✘       ✘      ✘
\          Integrated gradient   ✓        ✘            ✘       ✘     ✓       ✓      ✘
\          Counterfactual        ✓        ✘            ✘       ✘     ✘       ✓      ✘
\          Linear models         ✓        ✓            ✓       ✓     ✘       ✓      ✓
\          Tree models           ✓        ✓            ✓       ✓     ✘       ✓      ✓
\          L2X                   ✓        ✘            ✘       ✘     ✘       ✘      ✘
Image      LIME                  ✓        ✘            ✘       ✘     ✓       ✘      ✘
\          SHAP                  ✓        ✘            ✘       ✘     ✓       ✘      ✘
\          Integrated gradient   ✓        ✘            ✘       ✘     ✓       ✓      ✘
\          Grad-CAM, Grad-CAM++  ✓        ✘            ✘       ✓     ✓       ✘      ✘
\          Contrastive           ✓        ✘            ✓       ✘     ✘       ✓      ✘
\          Counterfactual        ✓        ✘            ✘       ✘     ✘       ✓      ✘
\          L2X                   ✓        ✘            ✘       ✘     ✘       ✘      ✘
Text       LIME                  ✓        ✘            ✘       ✓     ✓       ✘      ✘
\          SHAP                  ✓        ✘            ✘       ✘     ✓       ✘      ✘
\          Integrated gradient   ✓        ✘            ✘       ✘     ✓       ✓      ✘
\          L2X                   ✓        ✘            ✘       ✘     ✘       ✘      ✘
=========  ====================  =======  ===========  ======  ====  ======  =====  ========

Installation
############

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
###############

To get started, we recommend the linked tutorials in :ref:`Tutorials & Example Code <tutorial>`.
In general, we recommend using :py:mod:`omnixai.explainers.tabular.TabularExplainer`, :py:mod:`omnixai.explainers.vision.VisionExplainer`,
and :py:mod:`omnixai.explainers.nlp.NLPExplainer` for tabular, vision and NLP tasks, respectively. To generate explanations,
one only needs to specify

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
#################

Thank you for your interest in contributing to the library!
Before you get started, clone this repo, run ``pip install pre-commit``, and run ``pre-commit install`` from the root
directory of the repo. This will ensure all files are formatted correctly and contain the appropriate
license headers whenever you make a commit. To add a new explanation method into the library,
one may follow the steps below:

#. Choose the task type of the new explainer, e.g., "tabular", "vision" or "nlp".
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
#. Implement the "explain" function, e.g., ``explain(self, **kwargs)`` for global explanations, or
   ``explain(self, X, **kwargs)`` for local explanations where the type of ``X`` is class `Tabular`, `Image` or `Text`.
#. Import the explainer class in "__init__.py" of the packages :py:mod:`omnixai.explainers.tabular`,
   :py:mod:`omnixai.explainers.vision`, or :py:mod:`omnixai.explainers.nlp`.

The new explainer will be registered automatically, which can be called via :py:mod:`omnixai.explainers.tabular.TabularExplainer`,
:py:mod:`omnixai.explainers.vision.VisionExplainer` or :py:mod:`omnixai.explainers.nlp.NLPExplainer` by specifying one of the names
defined in ``alias``.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   omnixai
   tutorials


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
