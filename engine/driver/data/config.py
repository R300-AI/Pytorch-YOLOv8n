from .dataset import Detect_Dataset
import os

class Vision2D_Config():
    def __init__(self, args):
        '''
        依照專案資料夾的來源/目的地轉移的過程將步驟進行拆解：
            1. __init__：取得來源與目的地的位置。
            2. generate：執行轉移流程。
        '''
        self.args = args
        config_list = {'Detect': Detect_Dataset(self.args.FORMAT)}
        if self.args.TASK in config_list: 
            self.config = config_list[self.args.TASK]
        else:
            print('task_name', self.args.TASK, 'are not supportable, please check your configuration.')

        # 依照資源類型設定來源路徑與目標路徑
        self.target_path = os.getcwd() + '/datasets/{d}/{t}/{f}/{n}'.format(d=self.args.DATA_TYPE, t=self.args.TASK, f=self.args.FORMAT, n=self.args.PROJECT_NAME)
        if os.path.exists(self.target_path + '/{v}'.format(v=self.args.SOURCE)):
            # 若給定的資料來源為版本號，則自動搜尋同專案底下可用的其他版本號做為目標路徑
            self.source_path, v = os.getcwd() + '/datasets/{d}/{t}/{f}/{n}/{v}'.format(d=self.args.DATA_TYPE, t=self.args.TASK, f=self.args.FORMAT, n=self.args.PROJECT_NAME, v=self.args.SOURCE), 1
            while 'v' + str(v) in os.listdir(self.target_path): v += 1
            self.target_path += '/v' + str(v)
        else: 
            # 若給定的資料來源為資源路徑，則以這個路徑的資料集作為初始資料集重建專案
            self.source_path = os.getcwd() + self.args.SOURCE
            self.target_path += '/Benchmark'

    def generate(self):
        #檢查資料格式
        passed = self.config.CheckDataFormat(self.source_path)
        #建立系統路徑
        if passed == True:
            self.config.Generate_Directory(self.source_path, self.target_path, self.args)
            #執行資料擴增
            self.config.Augmentatation(self.target_path)
            #彙整資訊
            self.config.Summary(self.target_path)