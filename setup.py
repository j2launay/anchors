from setuptools import setup

setup(name='anchor_exp',
      version='0.0.1.0',
      description='Anchor explanations for machine learning models',
      url='http://github.com/marcotcr/anchor',
      author='Marco Tulio Ribeiro',
      author_email='marcotcr@gmail.com',
      license='BSD',
      packages=['anchor'],
      python_requires='>=3.5',
      install_requires=[
          'numpy',
          'scipy',
          'spacy',
          'csv',
          'emojis',
          'pandas',
          'sys',
          'time',
          'warnings',
          'utilities',
          'copy',
          'os',
          'matplotlib',
          'scikit-learn>=0.22'
      ],
      include_package_data=True,
      zip_safe=False)
