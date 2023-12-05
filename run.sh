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

###############################################
# 自動建立Docker執行環境，需滿足已下掛件：
# 1. Docker執行所需掛載的的資料集統一置於DATASET_DIR=./tmp/datasets/$USERNAME/$DATASET
# 2. Docker執行過程中的logs需動態寫入本機LOG_DIR=./tmp/logs/$USERNAME/$DATASET.log
# 3. Docker輸出的模型檔需包裝成Benchmark資料夾，並掛載於本機的OUTPUT_DIR=./tmp/outputs/$USERNAME/$DATASET
# 4. 其餘則依照engine所需自行掛載。
###############################################
ENGINE_DIR=$(dirname "$0")
DATASET_DIR=$ENGINE_DIR/tmp/datasets/$USERNAME/$DATASET
LOG_DIR=$ENGINE_DIR/tmp/logs/$USERNAME/$DATASET.log
OUTPUT_DIR=$ENGINE_DIR/tmp/outputs/$USERNAME/$DATASET

ENGINE_NAME=$(python3 -c "import json; data=json.load(open('${ENGINE_DIR}/spec.json')); print(data['name'].lower())")
CONTAINER_NAME=$(python3 -c "print('${USERNAME}/${DATASET}/${ENGINE_NAME}'.replace('/', '_'))")

echo "Start to build ${ENGINE_NAME} docker engine..."
#重置刷新原有的引擎映像檔
docker rmi -f $ENGINE_NAME     
docker rm -f $CONTAINER_NAME 
docker build -f $ENGINE_DIR/docker/Dockerfile . -t $ENGINE_NAME
echo "${ENGINE_NAME} docker engine has been build."

#重置刷新原有的log檔及路徑
mkdir $ENGINE_DIR/tmp
mkdir $ENGINE_DIR/tmp/logs
mkdir $ENGINE_DIR/tmp/outputs
mkdir $ENGINE_DIR/tmp/logs/$USERNAME
touch $LOG_DIR && > $LOG_DIR
rm -i -r -f $OUTPUT_DIR/*


###############################################
# 將資料集掛載於Docker環境，並開始進行訓練。
###############################################
echo "Start to run ${ENGINE_NAME} docker engine..."
docker container run --name $CONTAINER_NAME -it -v $OUTPUT_DIR:/usr/src/ultralytics/output -v $DATASET_DIR:/usr/src/ultralytics/dataset -v $ENGINE_DIR/engine:/usr/src/ultralytics/engine $ENGINE_NAME | tee -a $LOG_DIR
echo "Benchmark saved to ${OUTPUT_DIR}"
###############################################
# 清除訓練環境及暫存資源(研發期間暫不刪除)
###############################################
docker rmi -f $ENGINE
docker rm -f $CONTAINER     #刪除訓練環境，表示訓練結束
rm -i -r -f $DATASET_DIR/*
