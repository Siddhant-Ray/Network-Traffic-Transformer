#!/bin/bash

## First in arguments is the topology
## Second to fourth Specify the different congestion for different receivers (default is 0)
## Further arguments can be set later.

./docker-run.sh waf --run "trafficgen 
                    --topo=$2
                    --apps=20
                    --apprate=1Mbps
                    --startwindow=50 
                    --queuesize=1000p
                    --linkrate=30Mbps
                    --congestion1=$2Mbps
                    --congestion2=$3Mbps
                    --congestion3=$4Mbps
                    --prefix=results/large_test_disturbance_with_message_ids$5
                    --seed=$5"

