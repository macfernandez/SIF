from setuptools import setup

setup(
    name='sif_embedding',
    packages=['sif_embedding'], 
    install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'Theano',
        'Lasagne'
    ]
)