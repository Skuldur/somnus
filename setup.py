with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="somnus",
    version="0.0.1",
    author="Sigurður Skúli Sigurgeirsson",
    author_email="siggiskuli@gmail.com",
    description="Somnus allows you to listen for and detect a specific keyword in a continuous stream of audio data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Skuldur/somnus",
    packages=setuptools.find_packages(),
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