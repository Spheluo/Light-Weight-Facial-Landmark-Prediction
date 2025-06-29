# **shuffle_net_MSE_heatmap_code**

## 程式介紹
本程式的model是使用Shuffle Net v2 為基礎，使用MSE LOSS更新參數，最後是輸出68張heatmap作為輸出，再根據每一張heatmap最大值的座標作為landmark的座標
## 內含檔案
在同名的資料夾中包含以下檔案:
* [資料夾] models
  * Shuffle Net v2 model
* [資料夾] save_dir
  * 訓練好的models
* [py檔案] cfg.py
* [py檔案] main.py
* [py檔案] myDatasets.py
* [py檔案] eval.py
* [py檔案] tool.py

## 介紹程式(py檔案目的)
* myDatasets.py - training & validation資料整理與前處理
* tool.py - 主要的training函數及儲存models
* cfg.py - model training的相關參數與名稱
* main.py - 主程式，調用其他函式
* eval.py - 將model對所有test data預測結果保存在txt檔案中

## 如何運行
1.首先將myDatasets.py中第12行的兩個引數train_root & val_root 改為自己電腦中train & validation data的資料夾。
```sh
def get_train_val_set(train_root='data/synthetics_train/', val_root='data/aflw_val/'):
```
2.在main.py中第55行的兩個引數train_root & val_root 同1.改成對應資料夾。
```sh
train_set, val_set =  get_train_val_set(train_root='data/synthetics_train/', val_root='data/aflw_val/')
```
3.在cfg.py中可以改變model_type來改變model的名稱[對應在acc_log與save_dir中儲存model的資料夾名稱]，還有相對應的參數都可以改變，包含以下:
| 參數 | 意義 |
| ------ | ------ |
| batch_size | data 送進model的 batch size|
| lr| Learning rate |
| milestones | 哪些epoch後要降低Learning rate |
| num_epoch | epoch總數量 |

4.最後直接run main.py 即可開始training。

5.如果要輸出成上傳的格式，可以使用eval.py，將第59行的path改成存model的位置，並將第76行的dir_path改成test data的資料夾位置，隨後直接run此程式即可，完成後會發現在資料夾中會有一個result.txt的檔案，就是比賽需要的solution。
```sh
    path = 'save_dir/ShuffleNet/best_model16_4.pt'
```
```sh
    dir_path = 'data/aflw_test/'
```

