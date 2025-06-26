from setuptools import setup, find_packages

setup(
    name='av_ddm',       # Replace with your package name
    version='0.1.0',                # Initial release version
    description='Analysis tools to fir the drift diffusion model for an audiovisual task',
    author='Flora Takacs',            
    url='https://github.com/takacsflora/av-ddm',  # Link to your repository (optional)
    packages=find_packages(),   # Automatically find packages in the "src" folder
    install_requires=[                     # List of dependencies
        'numpy==1.21.5',
        'pandas==1.4', 
        'matplotlib',
        'ipykernel',
        'scikit-learn',
        'seaborn',
        "pyddm @ git+https://github.com/mwshinn/PyDDM.git@master"



    ],
    classifiers=[                          # Optional metadata for PyPI
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',               # Specify the minimum Python version
)