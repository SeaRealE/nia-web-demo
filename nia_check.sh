#!/bin/bash

count=1

while [ 1 ]
	do
		pid=`ps -ef | grep "NIA/app.py" | grep -v 'grep' | awk '{print $2}'`
		if [ -z $pid ] ; then
			echo "log) NO NIA_"$count >> /home/website/NIA/log_NIA.txt
			if [ $count -gt 3 ] ; then
				echo "log) NIA DEMO START" >> /home/website/NIA/log_NIA.txt
				count=0
				
				nohup python app.py &>> /home/website/NIA/log_NIA.txt &
		
			fi
		fi

		count=$((count+1))
		sleep 10
	done
