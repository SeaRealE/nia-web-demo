#!/bin/bash
sudo ufw allow 8893
python tools/load_model.py

PATH_NAME=$(pwd)
count=1
while [ 1 ]
	do
		pid=`ps -ef | grep "nia-web-demo/app.py" | grep -v 'grep' | awk '{print $2}'`
		if [ -z $pid ] ; then
			echo "log) NO NIA_"$count >> PATH_NAME'/nia-web-demo/log_NIA.txt'
			if [ $count -gt 3 ] ; then
				echo "log) NIA DEMO START" >> PATH_NAME'/nia-web-demo/log_NIA.txt'
				count=0
				nohup python app.py &>> PATH_NAME'/nia-web-demo/log_NIA.txt' &
			fi
		fi
		count=$((count+1))
		sleep 10
	done