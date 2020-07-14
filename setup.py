from setuptools import setup, find_packages

setup(
    name='pareto',
    version='0.1',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'torchvision',
        'tqdm',
    ],
)
