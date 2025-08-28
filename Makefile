# Makefile
IMAGE ?= v-jepa
TAG   ?= latest
NOCACHE ?= 0
PULL    ?= 1
PORT  ?= 8080
GPUS ?= all

# Used only when downloading a remote file in ovx build
RUN_FILE_URL ?= https://raw.githubusercontent.com/j3soon/omni-farm-isaac/master/scripts/docker/run.sh
RUN_FILE_PATH ?= /run.sh

DOCKER_BUILD = docker build $(if $(PULL),--pull,) $(if $(NOCACHE),--no-cache,)

.PHONY: base ovx

base:
	$(DOCKER_BUILD) --target base \
		-t $(IMAGE):$(TAG)-base \
		-f Dockerfile .

ovx:
	$(DOCKER_BUILD) --target ovx \
		--build-arg RUN_FILE_URL=$(RUN_FILE_URL) \
		--build-arg RUN_FILE_PATH=$(RUN_FILE_PATH) \
		-t $(IMAGE):$(TAG)-ovx \
		-f Dockerfile .

# Train: copy code inside image at build time, then run train.sh
train:
	docker run --rm -it \
		--gpus $(GPUS) -d \
		-p $(PORT):$(PORT) \
		-w /workspace \
		$(IMAGE):$(TAG)-base \
		bash ./train.sh

# Develop: mount current dir for live development
develop:
	docker run --rm -it \
		--gpus $(GPUS) -d \
		-p $(PORT):$(PORT) \
		-v $(PWD):/workspace \
		-w /workspace \
		$(IMAGE):$(TAG)-base \
		bash

clean:
	@-IMAGES=$$(docker images "$(IMAGE)" -q) ; \
	if [ -n "$$IMAGES" ]; then docker rmi -f $$IMAGES; fi
