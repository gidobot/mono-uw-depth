#! /bin/bash

docker run --rm -it \
	--net=host \
    --runtime=nvidia \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	--env=NVIDIA_DRIVER_CAPABILITIES=all \
	--env=DISPLAY \
	--env=QT_X11_NO_MITSHM=1 \
	--device=/dev/dri:/dev/dri \
	--pid=host \
	--cap-add=SYS_ADMIN \
	--cap-add=SYS_PTRACE \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v $PWD/../:/depth_estimation:rw \
	-v /media/gidobot/data/UWslam_dataset:/media/gidobot/data/UWslam_dataset \
	-v ~/.docker_bash_history:/root/.bash_history:rw \
	-w /depth_estimation \
	mono-uw-depth:latest \
	bash

#	--gpus all \
