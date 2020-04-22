"""Install command: pip3 install -e ."""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['pydicom', 'numpy', 'scipy', 'matplotlib']

setup(name='ecg_medical_research',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      description='Deep learning methods for 12-lead ECG signals classifications',
      url='http://github.com/tomergolany/ecg_medical_research',
      author='Tomer Golany',
      author_email='tomer.golany@gmail.com',
      license='Technion',
      packages=find_packages(),
      include_package_data=False,
      zip_safe=False)
