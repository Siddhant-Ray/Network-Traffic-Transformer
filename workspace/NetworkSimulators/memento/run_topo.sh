#!/bin/bash

## First in arguments is the topology
## Second to fourth Specify the different congestion for different receivers (default is 0)
## Further arguments can be set later.

./docker-run.sh waf --run "trafficgen_small 
                    --topo=$1
                    --apps=20
                    --apprate=1Mbps
                    --startwindow=50 
                    --queuesize=1000p
                    --linkrate=30Mbps
                    --congestion1=$2Mbps
                    --prefix=results/small_test_one_disturbance_with_message_ids11
                    --seed=11"

