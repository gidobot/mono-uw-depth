#! /bin/bash

docker build --network=host -f Dockerfile.jetsontorch -t mono-uw-depth:torch .
