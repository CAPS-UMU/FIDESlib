# FIDESlib Docker Images

This directory contains Docker images that enable compiling and using FIDESlib without the burden of dependency management.

This directory contains the following Dockerfiles:
* **Dockerfile.NVIDIA**: Use if your system have NVIDIA devices.

## Image building

> [!IMPORTANT]
> Image building will avoid downloading again the FIDESlib repository by doing a copy of the current one into the container image. To enable this, images must be build using the root directory of the project as working directory.

```bash
$pwd
/FIDESlib
$docker build -t <tag> -f docker/Dockerfile .
...
```

> [!IMPORTANT]
> Containers using this images will need to have access to the GPUs that you intent to use. As Docker by default does not expose them in the container, check the Docker documentation to see the methods that enable this functionality. You may need to be on the video group or administrator proviledges on the host system.

## Using FIDESlib

The building process of the container image copeid the FIDESlib repository into /root/fideslib. To use the library or run the test and benchmarks you must follow the compilation steps provided on the Readme at the repository root directory.