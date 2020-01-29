#!/bin/bash
mkdir -p data/UV_data
cd data/UV_data
wget https://dl.fbaipublicfiles.com/densepose/densepose_uv_data.tar.gz
tar xvf densepose_uv_data.tar.gz
rm densepose_uv_data.tar.gz
cd ../..
