#!/bin/bash

docker stop flask_test
docker rm flask_test
docker rmi -f kraken/flask_test
