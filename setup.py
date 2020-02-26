# python3 setup.py sdist bdist_wheel && python3 -m twine upload dist/*
from setuptools import setup
from setuptools import find_packages


version_py = "mustache/_version.py"
exec(open(version_py).read())

setup(
    name="mustache_hic",
    version=__version__,
    description='Mustache is a Hi-C analysis tool',
    long_description="Mustache is a Hi-C analysis tool",
    url='http://github.com/ay-lab/mustache/',
    entry_points={
        "console_scripts": ['mustache = mustache.mustache:main']
    },
    python_requires='>=3.5',
    author='Ferhat Ay',
    author_email='ferhatay@lji.org',
    license='MIT',
    packages=['mustache'],
    install_requires=[
        'cooler',
        'hic-straw',
        'numpy',
        'requests',
        'scipy',
        'pandas',
        'statsmodels',
        'pathlib',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    zip_safe=False,
)
