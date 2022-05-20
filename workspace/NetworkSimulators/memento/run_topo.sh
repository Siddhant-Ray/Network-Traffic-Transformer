#!/bin/bash

## First in arguments is the topology
## Second to fourth Specify the different congestion for different receivers (default is 0)
## Further arguments can be set later.

./docker-run.sh waf --run "trafficgen 
                    --topo=$1
                    --apps=20
                    --apprate=1Mbps
                    --startwindow=1 
                    --queuesize=100p
                    --linkrate=30Mbps
                    --congestion1=$2Mbps
                    --congestion2=$3Mbps
                    --congestion3=$4Mbps 
                    --prefix=results/topo_test_$1"

