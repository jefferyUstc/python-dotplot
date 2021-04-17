from pathlib import Path

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="python_dotplot",
    version="0.0.1b1",
    author='jefferyUstc',
    author_email="jeffery_cpu@163.com",
    description="light weighted dotplot drawer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jefferyUstc/python-dotplot",
    project_urls={
        'python-dotplot': 'https://github.com/jefferyUstc/python-dotplot',
    },
    zip_safe=False,
    keywords="dotplot python scatter",
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    install_requires=[
        path.strip() for path in Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    entry_points={},
    package_data={},
    include_package_data=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Natural Language :: English",
    ],
)
