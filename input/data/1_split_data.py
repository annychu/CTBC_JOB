# coding:utf-8

import os

'''
切分訓練資料集，
原始的 nlpcc-iccpol-2016.kbqa.training-data 有 14609 個樣本，
將其中12,009個複制成為訓練集(train.txt), 1,300個複制成驗證集(dev.text)、1,300個複制成測試集(test.txt)
'''

data_dir = 'NLPCC2016KBQA'
file_name = 'nlpcc-iccpol-2016.kbqa.training-data'

file_path_name = os.path.join(data_dir, file_name)
file = []
with open(file_path_name, 'r') as f:
    for line in f:
        line = line.strip()
        if line == '':
            continue
        file.append(line)
    f.close()

    assert len(file) % 4 == 0  # 每四列為一個樣本
    total_num = len(file) / 4  # 總筆數
    train_num = 12009 * 4  # 訓練集的列數
    dev_num = train_num + (1300 * 4)  # 驗證集的列數

    with open(os.path.join(data_dir, 'train.txt'), "w") as d:
        d.write('\n'.join(file[:train_num]))

    with open(os.path.join(data_dir, 'test.txt'), "w") as d:
        d.write('\n'.join(file[train_num:dev_num]))

    with open(os.path.join(data_dir, 'dev.txt'), "w") as d:
        d.write('\n'.join(file[dev_num:]))

print("Done")
