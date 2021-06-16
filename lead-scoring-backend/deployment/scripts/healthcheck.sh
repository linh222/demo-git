#!/usr/bin/env sh

# Healthcheck pods up and running before exiting deployment in 2 minutes

max_retries=60
retry_counter=0
APPLICATION=$PROJECT

while [ `kubectl get pod --no-headers --namespace=$KUBE_NAMESPACE|grep $APPLICATION |awk '{print $3}'|grep -vE "Running|Completed|Terminating|Evicted"|wc -l` != 0 ]
do
  if [ ${retry_counter} -eq ${max_retries} ]; then
      echo "Max attempts reached"
      exit 1
  fi

  echo "Waiting $APPLICATION pods up and running..."
  retry_counter=$(( $retry_counter + 1 ))
  sleep 5
done
