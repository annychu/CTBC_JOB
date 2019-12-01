# CTBC_JOB
## 第一題 NLP文本分類
### emotion_classifier.ipynb
### 處理流程說明:
1. 訓練資料預處理:

- 可處理的文字最大長度為384，所以每個review只取前384個字元。

- 訓練資料分成90:10的訓練和驗證集

2. 定義資料處理函式:

- 每次將csv裡的一筆資料轉成bert相容格式，並回傳:
tokens_tensor :: 句子，包含"[CLS]"
segments_tensor :: 皆為1，識別句子。
label_tensor :: 就tensor

3. 定義mini_batch函式:

- 將傳入的資料打包，input : sentimentDataset回傳值的集合，output : 
tokens_tensors :: (batch_size, max_seq_len_in_batch)
segments_tensors :: (batch_size, max_seq_len_in_batch)
masks_tensors :: (batch_size, max_seq_len_in_batch)  # 界定自注意力範圍，1是關注，0是padding 不需要關注。
label_tensors :: (batch_size)

4. 在BERT的SequenceClassification上再加上三層classifier，最後一層的輸出參數為2

5. 載入訓練資料，開始訓練模型，learning rate=1e-6, 跑20 epoches，每一個epoch 計算loss, accuracy，驗證模型，計算正確率。最後，儲存模型。

6. 定義一個可以input文字和可以oupput預測判斷的class

7. 讀入要預測的文字檔，整理格式，並將label先清除。

8. 載入模型，預測結果。

9. 將結果整理成原格式，並另存成csv。結束
