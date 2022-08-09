#!/bin/bash

# Original author: Siddhant Ray

## First in arguments is the topology
## Second to fourth Specify the different congestion for different receivers (default is 0)
## Last argument is for setting the seed for the random number generator (change for multiple runs).

## Current setup generates fine-tuning data with 4 bottlenecks, $2 is the second bottleneck rate
## $3 is the third bottleneck rate, $4 is the fourth bottleneck rate (all !=0)

## Topology is 
#    // Network topology 2
#    //
#    //                                             disturbance1
#    //                                                |
#    //           3x n_apps(senders) --- switchA --- switchB --- receiver1
#    //                      |
#    //                      |
#    //                      |        disturbance2
#    //                      |           |
#    //                   switchC --- switchD --- receiver2
#    //                                  |       
#    //                                  |                 
#    //                                  |                   disturbance3
#    //                                  |                       |
#    //                               switchE --- switchF --- switchG--recevier3
#    //

# If running inside the VSCode's environment to run Docker containers: Replace ./docker-run.sh waf with just waf 

mkdir -p results
./docker-run.sh waf --run "trafficgen 
                    --topo=$1
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

