#!/bin/sh

# Compile

make clean
make pump DIM=6 NUM=2000
./pump

