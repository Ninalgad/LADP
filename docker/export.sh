#!/usr/bin/env bash

./build.sh

docker save bondbidhie2023_algorithm | xz -T0 -c > bondbidhie2023_algorithm.tar.xz