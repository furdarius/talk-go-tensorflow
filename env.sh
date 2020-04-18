#!/bin/sh

export LIBRARY_PATH="$(pwd)/libtensorflow/lib"

# Linux only way to import .so (use DYLD_LIBRARY_PATH for MacOS)
export LD_LIBRARY_PATH=${LIBRARY_PATH}

# Headers
export C_INCLUDE_PATH="$(pwd)/libtensorflow/include"
