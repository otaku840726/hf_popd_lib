from setuptools import setup, find_packages

setup(
    name='popd',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'gradio[oauth]==5.12.0',
        'numpy==1.24.4',
        'Pillow==10.3.0',
        'Requests==2.31.0',
        'torch',
        'git+https://github.com/huggingface/transformers.git',
        'pandas',
        'torchvision',
        'accelerate',
        'qwen-vl-utils'
    ],
    include_package_data=True,
    description='My private library for OCR and authenticity detection',
    author='Aubing',
    author_email='your.email@example.com',
    url='https://github.com/otaku840726/hf_popd_lib',
)
