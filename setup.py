from setuptools import setup, find_packages

setup(
    name='hf_popd_lib',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'Pillow',
        'pandas',
        'requests'
    ],
    include_package_data=True,
    description='My private library for OCR and authenticity detection',
    author='Aubing',
    author_email='your.email@example.com',
    url='https://github.com/otaku840726/hf_popd_lib',
)
