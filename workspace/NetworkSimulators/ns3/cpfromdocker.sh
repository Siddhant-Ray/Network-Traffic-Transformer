#!/bin/bash

sudo docker cp 467f79c77706:/ns3/outputs .  # change CONTAINER ID as required
sudo chmod -R 757 outputs
sudo rm -r ../outputs
sudo mv outputs ../ 
