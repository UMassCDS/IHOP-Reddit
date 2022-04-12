from setuptools import setup, find_packages

setup(
    name="ihop",
    version="0.3.0",
    packages=find_packages(include=["ihop", "ihop.*"]),
    install_requires=[
        "dash==2.3.1",
        "dash-bootstrap-components==1.1.3",
        "dash-daq==0.5.0",
        "gensim==4.1.2",
        "joblib==1.1.0",
        "numpy==1.21.2",
        "pandas==1.3.4",
        "plotly==5.6.0",
        "pyspark==3.2.0",
        "pytimeparse==1.1.8",
        "scikit-learn==1.0.1",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
