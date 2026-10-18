
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:  
    """
    This function will return the list of requirements
    """
    requirements = [] 
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [r.replace("\n", "") for r in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements



setup(

    name="PhotocatalysisProject",
    version="0.1.0",
    author="REUBEN",
    author_email="solomonreuben1010@gmail.com",
    description="End-to-End ML for Predicting Novel Semiconductor Photocatalysts for Water Splitting Hydrogen Production",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # Specify the source directory
    install_requires=get_requirements("requirements.txt"),
)