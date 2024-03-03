# NextFramePred4Pong

## Overview
This repository hosts the code for PSYCH209 project that combines NaViT (a variant of the Vision Transformer) with a diffusion model to predict the next video frame in the game of Pong. The goal of this project is to explore the capabilities of Vision Transformers and diffusion models in understanding and predicting dynamic scenes, using the classic game of Pong as a testbed. With a scenario as simple as Pong, this project attempts to compare the learning process for physical principles manifested in videos to the cognitive process of infants acquiring physical laws of the world. By accurately predicting future frames, we aim to demonstrate the potential of these models in video prediction and game dynamics understanding, and shed light on how a vision transformer can elucidate the developmental trajectory in humans learning about the physical world. 


## Setting Up the Environment
To get started with this project, first clone the repository to your local machine:

`git clone https://github.com/sunnysjys/NextFramePred4Pong.git`

`cd NextFrmaePred4Pong`

After cloning the repository, you need to set up the project environment. We use Conda for environment management to ensure consistency across different setups.

1. Create the Environment: 
To create the Conda environment with all the necessary dependencies, run:

`conda env create -f environment.yml`

This command creates a new Conda environment named psych209 based on the specifications in the environment.yml file.

2. Activate the Environment: Before working on the project or running any scripts, activate the environment using:

`conda activate psych209`

3. Updating the Environment: If there are updates to the dependencies or if you're pulling changes that include an updated environment.yml, you can update the environment using:

`conda env update --file environment.yml --prune`

The --prune option ensures that any dependencies removed from environment.yml since your last update are also removed from the environment.