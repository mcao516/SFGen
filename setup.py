from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="SFGen",
    version="0.1.0",
    author="Meng Cao",
    author_email="meng.cao@mail.mcgill.ca",
    description="Check our paper 'Successor Features For Efficient Multi-Subject Controlled Text Generation'",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=required,
    python_requires=">=3.8",
)