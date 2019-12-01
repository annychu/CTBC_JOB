## 第四題: Open Domain Question Answering
### 處理流程說明:
1. 1_split_data.py:
-切分訓練資料集，原始的 nlpcc-iccpol-2016.kbqa.training-data 有 14609 個樣本，將其中12,009個複制成為訓練集(train.txt), 1,300個複制成驗證集(dev.text)、1,300個複制成測試集(test.txt)

2. 2-construct_dataset_ner.py:
-將分割後的訓練/驗證/測試資料集，建構成訓練NER模型的樣本集:將每個問題句子中的字元標註，符合實體的字串，起啟位元標註 'B-LOC'，其他位元標註 'I-LOC'，非實體字串則標註 'O'

3. 3-construct_dataset_attribute.py:
-將分割後的訓練/驗證/測試資料集，建構成訓練屬性相似度模型的樣本集:
每個樣本有六筆屬性資料，第0筆為正確的屬性標註為1, 另外隨機提取5筆錯誤的屬性標註為0
資料格式: 問題  屬性  label 

4. 5-triple_clean.py:
-將原始資料集重新整理成 entity, attribute, answer格式，預測答案時，可依 命名實體 列出其相關的屬性，再依相似度找出答案

5. NER_main.py:
-訓練NER(命名實體)提取的模型
-命名實體提取的模型
-引用程式: BERT_CRF.py, CRF_Model.py

6. SIM_main.py:
-訓練屬性相似度的模型

7. test_NER.py
-檢視NER模型的預測結果

8. test_SIM.py
-檢視屬性相似度模型的預測結果

9. 預測結果
-載入要預測的csv檔
-載入NER, SIM二個模型
-將資料逐一丟入模型找出命名實體後，再與triple_clean.csv資料集比對，
-若找出的屬性其詞語在句子之中，則直接回傳答案，
-若不在，再由屬性相似度模型決定最符合的答案
-將預測結果依question, answer 格式存成csv檔。結束
