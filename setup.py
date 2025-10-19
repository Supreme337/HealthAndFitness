from setuptools import find_packages,setup
from typing import List
def get_requirements()->List[str]:
    requirements_lst:List[str]=[]
    try:
        with open("requirements.txt",'r') as file:
            lines=file.readlines()
            for line in lines:
                requirement=line.strip()
                if requirement and requirement!="-e .":
                    requirements_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt not found")
    return requirements_lst                     

setup(
    name="HealthAndFitness",
    version="0.0.1",
    author="Harsh Malik",
    author_email="harshmalik034@gmail.com",
    packages=find_packages(),
    install_requirements=get_requirements()
    )