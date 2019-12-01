from BERT_CRF import BertCrf
from transformers import BertTokenizer
from NER_main import NerProcessor, statistical_real_sentences, flatten, CrfInputFeatures
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from sklearn.metrics import classification_report
import torch
import numpy as np
from tqdm import tqdm, trange

processor = NerProcessor()
tokenizer_inputs = ()
tokenizer_kwards = {'do_lower_case': False,
                    'max_len': 128,
                    'vocab_file': './input/config/bert-base-chinese-vocab.txt'}
tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BertCrf(config_name='./input/config/bert-base-chinese-config.json',
                num_tags=len(processor.get_labels()), batch_first=True)
model.load_state_dict(torch.load('./output/best_ner.bin', map_location=device))


model = model.to(device)

# features = torch.load(cached_features_file)
features = torch.load('./input/data/ner_data/cached_dev_128', map_location=device)

all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
all_label = torch.tensor([f.label for f in features], dtype=torch.long)
dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)

sampler = RandomSampler(dataset)
data_loader = DataLoader(dataset, sampler=sampler, batch_size=256)
loss = []
real_token_label = []
pred_token_label = []

for batch in tqdm(data_loader, desc="test"):
    model.eval()
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'tags': batch[3],
                  'decode': True,
                  'reduction': 'none'
                  }
        outputs = model(**inputs)
        # temp_eval_loss shape: (batch_size)
        # temp_pred : list[list[int]] 长度不齐
        temp_eval_loss, temp_pred = outputs[0], outputs[1]

        loss.extend(temp_eval_loss.tolist())
        pred_token_label.extend(temp_pred)
        real_token_label.extend(statistical_real_sentences(batch[3], batch[1], temp_pred))

loss = np.array(loss).mean()
real_token_label = np.array(flatten(real_token_label))
pred_token_label = np.array(flatten(pred_token_label))
assert real_token_label.shape == pred_token_label.shape
ret = classification_report(y_true=real_token_label, y_pred=pred_token_label, digits=6, output_dict=False)

print(ret)

# CRF_LABELS = ["O", "B-LOC", "I-LOC"]


#               precision    recall  f1-score   support
#
#            0   0.995127  0.994652  0.994889     10471
#            1   0.994971  0.989167  0.992060      1200
#            2   0.991035  0.992845  0.991939      6569
#
#     accuracy                       0.993640     18240
#    macro avg   0.993711  0.992221  0.992963     18240
# weighted avg   0.993643  0.993640  0.993641     18240
