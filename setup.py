from setuptools import setup

setup(name='anchors',
      version='0.0.1.0',
      description='Anchor explanations for machine learning models',
      url='https://github.com/juliendelaunay35000/anchors',
      author='Julien Delaunay',
      author_email='juliendelaunay35000@gmail.com',
      license='BSD',
      packages=['anchor'],
      python_requires='>=3.5',
      install_requires=[
          'numpy',
          'scipy',
          'spacy',
          'emoji',
          'pandas',
       	  'yellowbrick',
	  'mdlp-discretization',
          'matplotlib',
	  'nltk',
          'scikit-learn>=0.22'
      ],
      include_package_data=True,
      zip_safe=False)
