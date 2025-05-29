from setuptools import setup, find_packages

setup(
    name="fastapi-yolo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.8",
)
