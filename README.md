<p align="center"><img src=".github/NiaNetLogo.png" alt="NiaPy" title="NiaNet"/></p>

---
[![PyPI Version](https://img.shields.io/badge/pypi-v1.0.0-blue)](https://pypi.org/project/nianet/)
![PyPI - Python Version](https://img.shields.io/badge/python-3.8-blue)
[![Downloads](https://static.pepy.tech/badge/nianet)](https://pepy.tech/project/nianet)
[![GitHub license](https://img.shields.io/badge/license-MIT-green)](https://github.com/SasoPavlic/NiaNet/blob/main/LICENSE)

### Nature-Inspired Algorithm-driven Convolutional Autoencoder Architecture search: Empowered by High-Performance Computing (HPC)
<p align="center"><img src=".github/search-space.webp" alt="Search space" title="Discovering search space" width="50%" /></p>

### Description 📝

The proposed method NiaNetCAE attempts to pick hyperparameters and Convolutional autoencoder architecture that will result in a successful
encoding and decoding (minimal difference between input and output). NiaNetCAE uses the collection of algorithms available
in the library [NiaPy](https://github.com/NiaOrg/NiaPy) to navigate efficiently in waste search-space.

### What it can do? 👀

* **Construct novel CONV AE's architecture** using nature-inspired algorithms.
* **Selects the best hyperparameters** for the given dataset.
* It can be utilized for **any kind of dataset**, which has **3D images** values.
* Applied for depth estimation

Our example of the search space for the depth estimation problem:
* **Search space** is defined by:
  * **Architecture problem:**
    * x: layer step _(difference between input and output dimension)_
    * y: Number of layers _(architecture depth)_
  * **Hyperparameter problem:**
    * z: activation function
    * w: optimizer algorithm
* Total solutions:
  * x: 304
  * y: 304
  * z: 8
  * w: 6
  * x * y * z * w = 4.435.968 unique solutions
* NiaNetCAE can find the best solution by leveraging the power of nature-inspired algorithms.

### Installation ✅

Installing NiaNetCAE with pip3:

TODO: Publish it to PyPi

```sh
pip3 install nianetcae
```

### Documentation 📘

The purpose of this paper is to get an understanding of the NiaNetCAE approach.

**TODO - Future Journal:**
[NiaNetCAE for depth estimation]()

### Examples

Usage examples can be found [here](nianetcae/experiments). Currently, there is an example for finding the appropriate Convolutional Autoencoder for depth estimation on NYU2 Dataset.

### Getting started 🔨

##### Create your own example:

1. Replace the dataset in [data](data) folder.
2. Modify the parameters in [main_config.py](configs/main_config.yaml)
2. Adjust the dataloader logic in [dataloaders](nianetcae/dataloaders) folder.
3. Specify the search space in [conv_ae.py](nianetcae/models/conv_ae.py) from your problem domain.
3. Redesign the fitness function in [cae_run.py](nianetcae/cae_run.py) based on your optimization.

##### Changing dataset:

Once the dataset is changed, dataloaders needs to be modified to be able for forwarding new shape of data to models.


##### Specify the search space:

Set the boundaries of your search space with [conv_ae.py](nianetcae/models/conv_ae.py).

The following dimensions can be modified:

* x: layer step
* y: number of layers,
* z: activation function
* w: optimizer algorithm.

You can run the NiaNetCAE script once your setup is complete.

##### Running NiaNetCAE script with Docker:

```docker build --tag spartan300/nianet:cae . ```

```
docker run \
  --name=nianet-cae \
  -it \
  -v $(pwd)/logs:/app/nianetcae/logs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/configs:/app/configs \
  -w="/app" \
  --shm-size 8G \
  --gpus all spartan300/nianet:cae \
  python main.py
```

##### Running NiaNetCAE script with Poetry [help](https://github.com/python-poetry/poetry/issues/4231#issuecomment-1182766775):
1. Run the installation via ```poetry install ```
2. Then run the task with```poetry run poe autoinstall-torch-cuda```

##### Running NiaNetCAE script with HPC SLURM:

1. First build an image with docker (above example)
2. Docker push to Docker Hub: ```docker push username/nianetcae:latest```
3. SSH into a HPC Cluster via your access credentials
4. Create the following _run.sh_ script: ```cat > run.sh```
```
#!/bin/bash
## Running code on SLURM cluster
#SBATCH -J nianet-cae
#SBATCH -o nianet-cae-%j.out
#SBATCH -e nianet-cae-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --mem-per-gpu=32GB  # memory per GPU
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

singularity exec -e --pwd /app -B $(pwd)/logs:/app/logs,$(pwd)/data:/app/data,$(pwd)/configs:/app/configs --nv docker://spartan300/nianet:cae python main.py
```
1. Make script executable: ```chmod +x run.sh```
2. Make sure that you have the following folders in your current directory: logs, data, configs
3. Set folder permissions to 777: ```chmod -R 777 logs data configs```
4. Submit your script to a job scheduler: ```SBATCH run.sh```

### HELP ⚠️

**saso.pavlic@student.um.si**

## Acknowledgments 🎓

* NiaNetCAE was developed under the supervision of [prof. dr. Domenec Puig](https://scholar.google.es/citations?hl=en&user=2lmYVYAAAAAJ&view_op=list_works&sortby=pubdate)
  at [University Rovira i Virgili](https://www.urv.cat/en/).
* Together with [dr. Saddam Abdulwahab](https://scholar.google.es/citations?user=6YE5eu4AAAAJ&hl=en) at [University Rovira i Virgili](https://www.urv.cat/en/)


* This code is a fork of [NiaPy](https://github.com/NiaOrg/NiaPy). I am grateful that the authors chose to open-source
  their work for future use.

[comment]: <> (# Cite us)

[comment]: <> (Are you using NiaNet in your project or research? Please cite us!)

[comment]: <> (### Plain format)

[comment]: <> (```)

[comment]: <> (S. Pavlič, I. F. Jr, and S. Karakatič, “NiaNet: A framework for constructing Autoencoder architectures using nature-inspired algorithms,” in Annals of Computer Science and Information Systems, 2022, vol. 30, pp. 109–116. Accessed: Oct. 08, 2022. [Online]. Available: https://annals-csis.org/Volume_30/drp/192.html)

[comment]: <> (```)

[comment]: <> (### Bibtex format)

[comment]: <> (```)

[comment]: <> (    @article{NiaPyJOSS2018,)

[comment]: <> (        author  = {Vrban{\v{c}}i{\v{c}}, Grega and Brezo{\v{c}}nik, Lucija)

[comment]: <> (                  and Mlakar, Uro{\v{s}} and Fister, Du{\v{s}}an and {Fister Jr.}, Iztok},)

[comment]: <> (        title   = {{NiaPy: Python microframework for building nature-inspired algorithms}},)

[comment]: <> (        journal = {{Journal of Open Source Software}},)

[comment]: <> (        year    = {2018},)

[comment]: <> (        volume  = {3},)

[comment]: <> (        issue   = {23},)

[comment]: <> (        issn    = {2475-9066},)

[comment]: <> (        doi     = {10.21105/joss.00613},)

[comment]: <> (        url     = {https://doi.org/10.21105/joss.00613})

[comment]: <> (    })

[comment]: <> (```)

[comment]: <> (### RIS format)

[comment]: <> (```)

[comment]: <> (TY  - CONF)

[comment]: <> (TI  - NiaNet: A framework for constructing Autoencoder architectures using nature-inspired algorithms)

[comment]: <> (AU  - Pavlič, Sašo)

[comment]: <> (AU  - Jr, Iztok Fister)

[comment]: <> (AU  - Karakatič, Sašo)

[comment]: <> (T2  - Proceedings of the 17th Conference on Computer Science and Intelligence Systems)

[comment]: <> (C3  - Annals of Computer Science and Information Systems)

[comment]: <> (DA  - 2022///)

[comment]: <> (PY  - 2022)

[comment]: <> (DP  - annals-csis.org)

[comment]: <> (VL  - 30)

[comment]: <> (SP  - 109)

[comment]: <> (EP  - 116)

[comment]: <> (LA  - en)

[comment]: <> (SN  - 978-83-962423-9-6)

[comment]: <> (ST  - NiaNet)

[comment]: <> (UR  - https://annals-csis.org/Volume_30/drp/192.html)

[comment]: <> (Y2  - 2022/10/08/19:08:20)

[comment]: <> (L1  - https://annals-csis.org/Volume_30/drp/pdf/192.pdf)

[comment]: <> (L2  - https://annals-csis.org/Volume_30/drp/192.html)

[comment]: <> (```)

## License

This package is distributed under the MIT License. This license can be found online
at <http://www.opensource.org/licenses/MIT>.

## Disclaimer

This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it
at your own risk!
