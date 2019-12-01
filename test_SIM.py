from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from SIM_main import SimProcessor, SimInputFeatures, cal_acc
import torch
from tqdm import tqdm, trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = SimProcessor()
tokenizer_inputs = ()
tokenizer_kwards = {'do_lower_case': False,
                    'max_len': 128,
                    'vocab_file': './input/config/bert-base-chinese-vocab.txt'}
tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)

features = torch.load('./input/data/sim_data/cached_dev_128', map_location=device)

all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
all_label = torch.tensor([f.label for f in features], dtype=torch.long)
test_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)

bert_config = BertConfig.from_pretrained('./input/config/bert-base-chinese-config.json')
bert_config.num_labels = len(processor.get_labels())

model = BertForSequenceClassification(bert_config)
model.load_state_dict(torch.load('./output/best_sim.bin', map_location=device))
model = model.to(device)

test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=256)

total_loss = 0.  # loss 的總合
total_sample_num = 0  # 樣本總數目
all_real_label = []  # 記錄所有的真實標籤list
all_pred_label = []  # 記錄所有的預測標籤list

for batch in tqdm(test_dataloader, desc="testing"):
    model.eval()
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels': batch[3],
                  }
        outputs = model(**inputs)
        loss, logits = outputs[0], outputs[1]

        total_loss += loss * batch[0].shape[0]  # loss * 樣本個數
        total_sample_num += batch[0].shape[0]  # 記錄樣本個數

        pred = logits.argmax(dim=-1).tolist()  # 得到預測的label轉為list

        all_pred_label.extend(pred)  # 記錄預測的 label
        all_real_label.extend(batch[3].view(-1).tolist())  # 記錄真實的label
loss = total_loss / total_sample_num
question_acc, label_acc = cal_acc(all_real_label, all_pred_label)

print("loss", loss.item())
print("question_acc", question_acc)
print("label_acc", label_acc)

# test
# loss 0.0380166557431221
# question_acc 0.9498987078666687
# label_acc 0.9826409816741943

# dev
# loss 0.026128364726901054
# question_acc 0.9572441577911377
# label_acc 0.9926713705062866

# train
# loss 0.01614166982471943
# question_acc 0.9722089171409607
# label_acc 0.9953110814094543
