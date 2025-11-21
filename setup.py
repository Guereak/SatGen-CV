from setuptools import setup, find_packages

setup(
    name="roadgen-cv",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pillow",
    ],
)
