#!/bin/bash 

if [ "$1" == "fetch" ] 
then 
	sudo docker pull notspecial/ns-3-dev
	sudo docker run -i -t notspecial/ns-3-dev
elif [ "$1" == "newcontainer" ] 
then
	sudo docker run -i -t notspecial/ns-3-dev
elif [ "$1" == "shell" ]
then 
	sudo docker exec -it 467f79c77706 bash # change CONTAINER ID as required
else
	sudo docker start -ai 467f79c77706  # change CONTAINER ID as required	
fi	
