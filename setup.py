from setuptools import setup

setup(name='ee1480-positional_encoder',
      version='0.0.1',
      author='Rogério Guimarães',
      packages=['positional_encoder'],
      description='Models and utils forCaltech EE 148 class, Homework 3, Part 2',
      license='MIT',
      install_requires=[
            'torch',
            'torchtext',
            'portalocker>=2.0.0'
      ],
)