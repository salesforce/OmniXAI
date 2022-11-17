#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from setuptools import setup, find_namespace_packages

extras_require = {
    "plot": ["plotly>=4", "dash>=2.0", "dash_bootstrap_components>=1.0"],
    "vision": ["torch>=1.7", "torchvision>=0.8.1", "opencv-python-headless>=4.4"],
    "nlp": ["nltk>=3.4.5", "polyjuice_nlp"],
    "bentoml": ["bentoml>=1.0.0"]
}
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name="omnixai",
    version="1.2.3",
    author="Wenzhuo Yang, Hung Le, Tanmay Shivprasad Laud, Silvio Savarese, Steven C.H. Hoi",
    description="OmniXAI: An Explainable AI Toolbox",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="XAI Explainable AI Explanation",
    url="https://github.com/salesforce/omnixai",
    license="3-Clause BSD",
    packages=find_namespace_packages(include="omnixai.*"),
    package_dir={"omnixai": "omnixai"},
    package_data={"omnixai": ["visualization/assets/*"]},
    install_requires=[
        "numpy>=1.17",
        "pandas>=1.1.0",
        "scikit-learn>=0.24,<1.2",
        "scipy>=1.5.0",
        "scikit-image>=0.17.2",
        "matplotlib",
        "pillow<10.0",
        "lime",
        "shap>=0.40.0",
        "SALib",
        "hnswlib>=0.5.0",
        "dill",
        "tqdm",
        "wheel",
        "packaging",
        "ipython",
        "tabulate",
        "statsmodels>=0.10.1"
    ],
    extras_require=extras_require,
    python_requires=">=3.7,<4",
    zip_safe=False,
)
