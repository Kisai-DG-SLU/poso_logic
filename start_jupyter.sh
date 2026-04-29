#!/bin/bash
cd /mnt/prod && pixi run jupyter lab --ip=0.0.0.0 --no-browser --allow-root
