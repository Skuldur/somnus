import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="somnus",
    version="0.1.2",
    author="Sigurður Skúli Sigurgeirsson",
    author_email="siggiskuli@gmail.com",
    description="Somnus allows you to listen for and detect a specific keyword in a continuous stream of audio data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Skuldur/somnus",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "numpy==1.16.2",
        "pydub==0.23.1",
        "pyaudio==0.2.11",
        "librosa==0.8.0",
        "tensorflow==2.2.0",
        "fire==0.3.1",
        "tqdm==4.47.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points = {
        'console_scripts': ['somnus=cli.cli:main'],
    },
    python_requires='>=3.6',
)
