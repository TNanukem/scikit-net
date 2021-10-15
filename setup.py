import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="scikit-net",
    version="0.0.2",
    description="Machine Learning in Complex Networks",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/TNanukem/scikit-net",
    download_url='https://github.com/TNanukem/scikit-net/archive/refs/tags/v0.0.1.tar.gz',  # noqa: E501
    keywords=['Machine Learning', 'Complex Networks'],
    author="Tiago Toledo Jr",
    author_email="tiago.nanu@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=['attrs',
                      'decorator',
                      'importlib-metadata',
                      'iniconfig',
                      'joblib',
                      'networkx',
                      'numpy',
                      'packaging',
                      'pandas',
                      'pluggy',
                      'py',
                      'pyparsing',
                      'pytest',
                      'python-dateutil',
                      'pytz',
                      'scikit-learn',
                      'scipy',
                      'six',
                      'sklearn',
                      'threadpoolctl',
                      'toml',
                      'tqdm',
                      'typing-extensions',
                      'zipp'],
)
