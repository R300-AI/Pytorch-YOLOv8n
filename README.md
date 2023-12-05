### 目錄結構
```
Pytorch-YOLOv8n
    ├─ docker
        ├─ Dockerfile                              # 依照不同硬體架構可能會需要改版(需在Dockerfile檔名後方加入_<硬體>，如：Dockerfile_GPU)，但預設就是直接安裝Dockerfie的版本
    ├─ engine                                      # Docker建構過程中所需搭配的程式庫，除非要調整引擎的工作細節，否則無需理會
        ├─    ... ...                              
    ├─ tmp                                         # 系統的暫存空間，當引擎被執行後會自動建立。
        ├─ datasets/<username>/<dataset>/          #使用者(username)選定的資料集，後端可透過Xdriver APIs自動pull進來(倘若此模組的放置位置正確了話)
            ├─      ... ...
        ├─ logs/<username>/<dataset>.log           #使用者(username)在執行run.sh過程中所產出的logs內容
        ├─ putputs/<username>/<dataset>/                #使用者(username)在執行完run.sh最終產生的結果皆存放與此路徑
            ├─      ... ...
    ├─ spec.json         # 引擎輸入/輸出的規格資訊，包括輸入資料的版本(version [str])、型態(dtype [str])、任務類型(task [str])、支援的輸出精度與格式(outputs [list])
    └─ run.sh            # 引擎的基本元素-(2)，需投入必要參數以執行引擎的主要訓練功能
```
