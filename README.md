# Generalized End-To-End Loss For Speaker Verification

This is my attempt at implementing the paper [Generalized End-To-End Loss For Speaker Verification](https://arxiv.org/pdf/1710.10467.pdf)

Still getting started with ML/Python/Docker, but I figured a good way to get up to speed would be to just start implementing papers that seemed interesting to me.

To get started locally with Docker first make a volume to store data/models in:

`docker create volume gen-e2e-sv`

To build the image, cd to the root of this repository, and run:

`docker build . -t d18n/gen-e2e-sv`

Then, run the following command, which will spin up the container and start a jupyter server on port 8888

`docker run -it --gpus all -p 8888:8888 -v gen-e2e-sv:/workspace/ d18n/gen-e2e-sv`

Disclaimer: Much of this code was adapted from reading the paper, and then using https://github.com/CorentinJ/Real-Time-Voice-Cloning as a reference
I can't take much credit for this implementation, but I hope to iterate on it at least a little bit