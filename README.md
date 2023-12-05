### 目錄結構
```
Pytorch-YOLOv8n
    ├─ docker
        ├─ Dockerfile                              # 依照不同硬體架構可能會需要改版(需在Dockerfile檔名後方加入_<硬體>，如：Dockerfile_GPU)，但預設就是直接安裝Dockerfie的版本
    ├─ engine                                      # Docker建構過程中所需搭配的程式庫，除非要調整引擎的工作細節，否則無需理會
        ├─    ... ...                              
    ├─ tmp                                         # 系統的暫存路徑，必須需符合以下規範：
        ├─ datasets/<username>/<dataset>/...       #使用者(username)選定的資料集，需透過後端Xdriver APIs自動pull進來
        ├─ logs/<username>/<dataset>.log           #使用者(username)在執行run.sh過程中所產出的logs內容
        ├─ putputs/<username>/<dataset>/ ...       #使用者(username)在執行完run.sh後，最終產生的檔案皆存放與此路徑
    ├─ spec.json         # 系統的配置檔，必須包含以下資訊：型態(dtype [str])、任務類型(task [str])、引擎名稱(name [str] *與repo相同, 但需將-換成/)、輸出精度與格式(outputs [list])
    └─ run.sh            # 引擎的基本元素-(2)，需投入必要參數以執行引擎的主要訓練功能
```
