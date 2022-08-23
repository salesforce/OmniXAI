omnixai.explainers package
==========================

This package contains all the supported explainers. In practice, we recommend using
use :py:mod:`omnixai.explainers.tabular.TabularExplainer`, :py:mod:`omnixai.explainers.vision.VisionExplainer`,
:py:mod:`omnixai.explainers.nlp.NLPExplainer`, and :py:mod:`omnixai.explainers.timeseries.TimeseriesExplainer`
for tabular, vision, NLP and time series tasks, respectively, and using :py:mod:`omnixai.explainers.data.DataAnalyzer`
and :py:mod:`omnixai.explainers.prediction.PredictionAnalyzer` for feature analysis and prediction result analysis.
To generate explanations, one only needs to specify the ML model, the pre-processing function (converting raw data into the model
inputs), the post-processing function (e.g., converting the model outputs into class probabilities, optional),
and the explainer names (e.g., lime, shap, gradcam):

.. code-block:: python

    explainers = TabularExplainer(
        explainers=["lime", "shap", "mace", "pdp"],        # The explainers to apply
        mode="classification",                             # The task type
        data=data,                                         # The data for initializing the explainers
        model=model,                                       # The ML model to explain
        preprocess=preprocess_function,                    # The preprocessing function
        postprocess=None,                                  # The postprocessing function
        params={
            "lime": {"kernel_width": 3},
            "shap": {"nsamples": 100},
            "mace": {"ignored_features": ["Sex", "Race", "Relationship", "Capital Loss"]}
        }                                                  # Additional parameters
    )
    local_explanations = explainers.explain(x)             # Generate local explanations given input `x`
    global_explanations = explainers.explain_global()      # Generate global explanations

For vision tasks, e.g., image classification, the explanations can be generated in a similar way:

.. code-block:: python

    explainer = VisionExplainer(
        explainers=["gradcam", "lime", "ig"],              # The explainers to apply
        mode="classification",                             # The task type
        model=model,                                       # The ML model to explain
        preprocess=preprocess_function,                    # The preprocessing function
        postprocess=postprocess_function,                  # The postprocessing function
        params={"gradcam": {"target_layer": model.layer4[-1]}}  # Additional parameters
    )
    local_explanations = explainer.explain(img)            # Generate local explanations given input `img`

For a single explainer, its constructor has one of the following two formats:

- **__init__(self, predict_function, **kwargs)**: This is for model-agnostic explainers. ``predict_function``
  is the prediction function of the black-box ML model to explain. The inputs of ``predict_function`` are the raw
  input features, and the outputs of ``predict_function`` are the model outputs.
- **__init__(self, model, preprocess_function, postprocess_function, **kwargs)**: This is for model-specific explainers.
  ``model`` is the ML model to explain. The model-specific explainers require
  some information about ``model``, e.g., whether ``model`` is differentiable (PyTorch or Tensorflow). ``preprocess_function``
  is the pre-processing function for ``model``, converting the raw features into the inputs of ``model``, e.g., resizing
  images to (224, 224) and normalizing pixel values. ``postprocess_function`` is the post-processing function for ``model``,
  which is used to convert the output logits into class probabilities. ``postprocess_function`` is mainly used for
  visualization, e.g., showing the predicted probabilities in the dashboard. We recommend:

  - ``model``: It outputs logits for classification or estimated values for regression given the pre-processed inputs.
  - ``preprocess_function``: It converts the raw input features into the model inputs.
  - ``postprocess_function``: It is ignored in regression tasks or converts logits into class probabilities
    in classification tasks.

For example, a model-agnostic explainer can be applied as follows:

.. code-block:: python

    # Load dataset
    data = np.genfromtxt('adult.data', delimiter=', ', dtype=str)
    feature_names = [
        "Age", "Workclass", "fnlwgt", "Education",
        "Education-Num", "Marital Status", "Occupation",
        "Relationship", "Race", "Sex", "Capital Gain",
        "Capital Loss", "Hours per week", "Country", "label"
    ]
    tabular_data = Tabular(
        data,
        feature_columns=feature_names,
        categorical_columns=[feature_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]],
        target_column='label'
    )
    # Preprocessing
    transformer = TabularTransform().fit(tabular_data)
    x = transformer.transform(tabular_data)
    # Train an XGBoost model
    gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
    gbtree.fit(x[:, :-1], x[:, -1])       # The last column in `x` is the label column

    # Construct the prediction function
    predict_function=lambda z: gbtree.predict_proba(transformer.transform(z))
    # Initialize the SHAP explainer
    explainer = ShapTabular(
        training_data=tabular_data,
        predict_function=predict_function
    )

For a model-specific explainer, e.g., Grad-CAM, one can apply the followings:

.. code-block:: python

    # A ResNet model to explain
    model = models.resnet50(pretrained=True)
    # Construct the prediction function
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    preprocess = lambda images: torch.stack([transform(im.to_pil()) for im in images])
    # Initialize the Grad-CAM explainer
    explainer = GradCAM(
        model=model,
        preprocess_function=preprocess,
        target_layer=model.layer4[-1]
    )

More detailed examples can be found in the tutorials.

.. automodule:: omnixai.explainers
   :members:
   :undoc-members:
   :show-inheritance:

omnixai.explainers.base module
------------------------------

.. automodule:: omnixai.explainers.base
   :members:
   :undoc-members:
   :show-inheritance:

Explainers for different tasks
------------------------------

.. toctree::
   :maxdepth: 2

   omnixai.explainers.data
   omnixai.explainers.prediction
   omnixai.explainers.tabular
   omnixai.explainers.vision
   omnixai.explainers.nlp
   omnixai.explainers.timeseries
   omnixai.explainers.ranking
