from distutils.core import setup

setup(
    name='NiaNetCAE',
    version='1.0.0',
    packages=['nianetcae', 'nianetcae.models', 'nianetcae.storage', 'nianetcae.visualize', 'nianetcae.dataloaders',
              'nianetcae.experiments', 'nianetcae.niapy_extension'],
    url='https://github.com/SasoPavlic/NiaNetCAE/tree/9419bb55dfa7d5a79f4e3824b727265dac07911e',
    license='MIT License',
    author='Saso Pavlic',
    author_email='saso.pavlic@student.um.si',
    description='Designing and constructing neural network topologies using nature-inspired algorithms - powered by high performance computer (HPC)'
)
