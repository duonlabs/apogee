from setuptools import setup, find_packages

setup(
    name='apogee',
    version='0.0.1',
    description='Establisshing Scaling Laws for Crypto Market Forecasting',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Duon Labs',
    author_email='apogee@duonlabs.com',
    url='https://github.com/duonlabs/apogee',
    packages=find_packages(),
    install_requires=[
        'requests',
        'numpy',
        'pandas',
        'torch',
        'huggingface_hub',
        'tqdm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)