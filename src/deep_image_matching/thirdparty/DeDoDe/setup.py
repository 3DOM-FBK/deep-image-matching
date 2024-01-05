from setuptools import setup, find_packages


setup(
    name="DeDoDe",
    packages=find_packages(include= ["DeDoDe*"]),
    install_requires=open("requirements.txt", "r").read().split("\n"),
    python_requires='>=3.9.0',
    version="0.0.1",
    author="Johan Edstedt",
)
