from setuptools import setup, find_packages

setup(
    name='ihop',
    version='0.1.0',
    packages=find_packages(include=['ihop', 'ihop.*']),
    install_requires=[
        'gensim==4.1.2',
        'numpy==1.21.2',
        'pandas==1.3.4',
        'pyspark==3.2.0',
        'scikit-learn==1.0.1'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)