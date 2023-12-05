# Pytorch-YOLOv8n
"""
├─ logs
    ├─ Logs.log      # 執行run.sh過程中會動態寫入執行的資訊，目前先以"finished"作為結束的token
├─ docker
    ├─ Dockerfile    # 依照不同硬體架構可能會需要改版(需在Dockerfile檔名後方加入_<硬體>，如：Dockerfile_GPU)，但預設就是直接安裝Dockerfie的版本
├─ engine
    ├─    ... ...    # Docker建構過程中所需搭配的程式庫，除非要調整引擎的工作細節，否則無需理會
├─ data.json         # 引擎輸入/輸出的規格資訊，包括輸入資料的版本(version [str])、型態(dtype [str])、任務類型(task [str])、支援的輸出精度與格式(outputs [list])
├─ install.sh        # 引擎的基本元素-(1)，用於安裝引擎的docker環境
└─ run.sh            # 引擎的基本元素-(2)，需投入必要參數以執行引擎的主要訓練功能
"""
