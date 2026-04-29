# Base image.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl

# Copy FIDESlib into the container.
ADD . /root/fideslib
ENV WORKDIR /root
RUN mkdir -p ${WORKDIR}
WORKDIR ${WORKDIR}