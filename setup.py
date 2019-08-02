import setuptools

setuptools.setup(
    name="rxntorch",
    version="0.0",
    description="Package for training graph-convolutional networks for reaction prediction.",
    author="John Herr",
    author_email="johnherr@gmail.com",
    url="https://github.com/jeherr/rxntorch",
    long_description=open('README.rst').read(),
    packages=setuptools.find_packages(),
    install_requires=[],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                'Operating System :: POSIX',
                'Programming Language :: Python',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.6'])
