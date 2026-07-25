"""Setup configuration for the anchors package."""
from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='anchors',
    version='0.0.1.0',
    description='Anchor explanations for machine learning models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/j2launay/anchors',
    author='Julien Delaunay',
    author_email='juliendelaunay35000@gmail.com',
    license='BSD',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.18',
        'scipy>=1.4',
        'spacy>=2.2',
        'emoji>=1.0',
        'pandas>=1.0',
        'yellowbrick>=1.1',
        'mdlp-discretization>=0.3',
        'matplotlib>=3.1',
        'nltk>=3.4',
        'scikit-learn>=0.22',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    include_package_data=True,
    zip_safe=False,
)
