from setuptools import setup, find_packages

# Read requirements.txt and use it for the install_requires parameter
with open("./requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="fast-fit",
    version="1.0.1",
    description="Fast and effective approach for few shot with many classes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/fastfit",
    author="Elron Bandel & Asaf Yehudai",
    author_email="elron.bandel@ibm.com",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Text Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="text classification machine learning NLP",
    packages=find_packages(
        exclude=[
            "contrib",
            "docs",
            "tests*",
            "exp*",
            "figures",
            "logs",
            "scripts",
            "tmp",
        ]
    ),
    install_requires=required,
    extras_require={
        "dev": ["check-manifest", "pytest"],
        "test": ["coverage"],
    },
    entry_points={
        "console_scripts": [
            "train_fastfit=fastfit.train:main",
        ],
    },
    project_urls={
        "Source": "https://github.com/IBM/fastfit/",
    },
)
