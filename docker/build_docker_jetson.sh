#! /bin/bash

docker build --network=host -f Dockerfile.jetson -t mono-uw-depth:latest .