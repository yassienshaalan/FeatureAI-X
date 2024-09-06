#!/bin/bash
. "/opt/spark/bin/load-spark-env.sh"
# When the spark work_load is master run class org.apache.spark.deploy.master.Master
if [ "$SPARK_WORKLOAD" == "master" ];
then

export SPARK_MASTER_HOST=`hostname`


cd /opt/spark/sbin && ./start-connect-server.sh --packages org.apache.spark:spark-connect_2.12:$SPARK_VERSION --host 0.0.0.0 --port 15002 --master spark://spark-master:7077
cd /opt/spark/sbin && ./start-history-server.sh

cd /opt/spark/bin && ./spark-class org.apache.spark.deploy.master.Master \
    --ip $SPARK_MASTER_HOST \
    --port $SPARK_MASTER_PORT \
    --webui-port $SPARK_MASTER_WEBUI_PORT >> $SPARK_MASTER_LOG
#    --properties-file /opt/spark/conf/spark-defaults.conf \

elif [ "$SPARK_WORKLOAD" == "worker" ];
then
# When the spark work_load is worker run class org.apache.spark.deploy.master.Worker
cd /opt/spark/bin && ./spark-class org.apache.spark.deploy.worker.Worker --webui-port $SPARK_WORKER_WEBUI_PORT $SPARK_MASTER >> $SPARK_WORKER_LOG

elif [ "$SPARK_WORKLOAD" == "submit" ];
then
    echo "SPARK SUBMIT"
else
    echo "Undefined Workload Type $SPARK_WORKLOAD, must specify: master, worker, submit, connect"
fi
