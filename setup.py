import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

with open(HERE / 'requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="scikit-net",
    version="0.0.1",
    description="Machine Learning in Complex Networks",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/TNanukem/sknet",
    download_url='https://github.com/TNanukem/sknet/archive/refs/tags/v0.0.1.tar.gz',
    keywords=['Machine Learning', 'Complex Networks'],
    author="Tiago Toledo Jr",
    author_email="tiago.nanu@gmailcom",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=required,
)
