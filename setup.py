from setuptools import setup, find_packages

test_deps = ["pytest"]
dev_deps = ["black"]

extras = {"test": test_deps, "dev": dev_deps}

setup(
    name="ihop",
    version="1.0.0",
    packages=find_packages(include=["ihop", "ihop.*", "ihop.*.*"]),
    install_requires=[
        "gensim==4.1.2",
        "joblib==1.1.0",
        "numpy==1.21.2",
        "matplotlib==3.5.0",
        "pandas==1.3.4",
        "plotly==5.6.0",
        "pyspark==3.2.0",
        "pytimeparse==1.1.8",
        "scikit-learn==1.0.1",
    ],
    tests_require=test_deps,
    extras_require=extras,
    include_package_data=True,
)
