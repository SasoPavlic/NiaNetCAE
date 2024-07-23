from distutils.core import setup

setup(
    name='NiaNetCAE',
    version='1.0.0',
    packages=['nianetcae', 'nianetcae.models', 'nianetcae.storage', 'nianetcae.visualize', 'nianetcae.dataloaders',
              'nianetcae.experiments', 'nianetcae.niapy_extension'],
    url='https://github.com/SasoPavlic/NiaNetCAE',
    license='MIT License',
    author='Saso Pavlic',
    author_email='saso.pavlic@student.um.si',
    description='Designing and constructing neural network topologies using nature-inspired algorithms - powered by high performance computer (HPC)'
)
