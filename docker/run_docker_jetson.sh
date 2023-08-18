#! /bin/bash

docker run --rm -it \
	--net=host \
	--runtime nvidia \
	-e DISPLAY=$DISPLAY \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v $PWD/../:/depth_estimation:rw \
	-v ~/.docker_bash_history:/root/.bash_history:rw \
	-w /depth_estimation \
	mono-uw-depth:latest \
	bash
