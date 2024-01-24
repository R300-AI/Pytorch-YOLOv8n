#!/bin/bash
###############################################
#設定系統參數，呼叫時需投入-u(Username) -d(Dataset)兩個參數以據此分離系統環境
###############################################
while getopts u:d: flag
do
    case "${flag}" in
        u) USERNAME=${OPTARG};;
        d) DATASET=${OPTARG};;
    esac
done

echo ""
echo "USERNAME $USERNAME"
echo "DATASET $DATASET"
echo ""

###############################################
# 自動建立Docker執行環境，需滿足已下掛件：
# 1. Docker執行所需掛載的的資料集統一置於DATASET_DIR=./tmp/datasets/$USERNAME/$DATASET
# 2. Docker執行過程中的logs需動態寫入本機LOG_DIR=./tmp/logs/$USERNAME/$DATASET.log
# 3. Docker輸出的模型檔需包裝成Benchmark資料夾，並掛載於本機的OUTPUT_DIR=./tmp/outputs/$USERNAME/$DATASET
# 4. 其餘則依照engine所需自行掛載。
###############################################
ENGINE_DIR=$(dirname "$0")
DATASET_DIR=$ENGINE_DIR/tmp/datasets/$USERNAME/$DATASET
OUTPUT_DIR=$ENGINE_DIR/tmp/outputs/$USERNAME/$DATASET

mkdir $ENGINE_DIR/tmp/logs
mkdir $ENGINE_DIR/tmp/logs/$USERNAME
LOG_DIR=$ENGINE_DIR/tmp/logs/$USERNAME/$DATASET.log
touch $LOG_DIR && > $LOG_DIR

echo ""
echo "ENGINE_DIR $ENGINE_DIR"
echo "LOG_DIR $LOG_DIR"
echo ""

ENGINE_NAME=$(python3 -c "import json; data=json.load(open('${ENGINE_DIR}/spec.json')); print(data['name'].lower())")
CONTAINER_NAME=$(python3 -c "print('${USERNAME}/${DATASET}/${ENGINE_NAME}'.replace('/', '_'))")

echo "Start to build ${ENGINE_NAME} docker engine..."
#重置刷新原有的引擎映像檔
docker rmi -f $ENGINE_NAME     
docker rm -f $CONTAINER_NAME 
docker build -f $ENGINE_DIR/docker/Dockerfile . -t $ENGINE_NAME
echo "${ENGINE_NAME} docker engine has been build."

###############################################
# 將資料集掛載於Docker環境，並開始進行訓練。
###############################################
echo "Start to run ${ENGINE_NAME} docker engine..."
docker container run --name $CONTAINER_NAME -it -v $OUTPUT_DIR:/usr/src/ultralytics/benchmark -v $ENGINE_DIR/engine:/usr/src/ultralytics/engine -v $DATASET_DIR:/usr/src/ultralytics/engine/dataset $ENGINE_NAME | tee -a $LOG_DIR
echo "Benchmark saved to ${OUTPUT_DIR}"
###############################################
# 清除訓練環境及暫存資源
###############################################
docker rmi -f $ENGINE_NAME
docker rm -f $CONTAINER_NAME
rm -r -f $DATASET_DIR/*
