from setuptools import setup, find_packages

with open('README.md', 'r') as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name="colocalisations",
    python_requires='>3.6',
    version="0.0.6",
    author="Aurélien COLIN",
    author_email="acolin@groupcls.com",
    description="Utilities to colocate various info to Sentinel1 products",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="n/a",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    package_data={
        'colocalisations': ['res/colormap/IR4AVHRR6.cpt', 'res/nexrad_stations.txt'],
    },
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ]
)
