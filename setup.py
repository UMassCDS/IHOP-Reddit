from setuptools import setup, find_packages

setup(
    name='ihop',
    version='0.0.0',
    packages=find_packages(include=['ihop', 'ihop.*']),
    install_requires=[
        'pandas==1.3.4',
        'pyspark==3.2.0'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)