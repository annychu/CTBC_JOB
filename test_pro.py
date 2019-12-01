from BERT_CRF import BertCrf
from NER_main import NerProcessor, CRF_LABELS
from SIM_main import SimProcessor, SimInputFeatures
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
import pandas as pd
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_ner_model(config_file, pre_train_model, label_num=2):
    model = BertCrf(config_name=config_file, num_tags=label_num, batch_first=True)
    model.load_state_dict(torch.load(pre_train_model))
    return model.to(device)


def get_sim_model(config_file, pre_train_model, label_num=2):
    bert_config = BertConfig.from_pretrained(config_file)
    bert_config.num_labels = label_num
    model = BertForSequenceClassification(bert_config)
    model.load_state_dict(torch.load(pre_train_model))
    return model


def get_entity(model, tokenizer, sentence, max_len=128):
    pad_token = 0
    sentence_list = list(sentence.strip().replace(' ', ''))
    text = " ".join(sentence_list)
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        truncate_first_sequence=True  # We're truncating the first sequence in priority if True
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)
    padding_length = max_len - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)
    labels_ids = None

    assert len(input_ids) == max_len, "Error with input length {} vs {}".format(len(input_ids), max_len)
    assert len(attention_mask) == max_len, "Error with input length {} vs {}".format(len(attention_mask), max_len)
    assert len(token_type_ids) == max_len, "Error with input length {} vs {}".format(len(token_type_ids), max_len)

    input_ids = torch.tensor(input_ids).reshape(1, -1).to(device)
    attention_mask = torch.tensor(attention_mask).reshape(1, -1).to(device)
    token_type_ids = torch.tensor(token_type_ids).reshape(1, -1).to(device)
    labels_ids = labels_ids

    model = model.to(device)
    model.eval()
    # 由於傳入的tag為None，所以返回的loss 也是None
    ret = model(input_ids=input_ids,
                tags=labels_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
    pre_tag = ret[1][0]
    assert len(pre_tag) == len(sentence_list) or len(pre_tag) == max_len - 2

    pre_tag_len = len(pre_tag)
    b_loc_idx = CRF_LABELS.index('B-LOC')
    i_loc_idx = CRF_LABELS.index('I-LOC')
    o_idx = CRF_LABELS.index('O')

    if b_loc_idx not in pre_tag and i_loc_idx not in pre_tag:
        print("没有在句子[{}]中找到實體".format(sentence))
        return ''
    if b_loc_idx in pre_tag:

        entity_start_idx = pre_tag.index(b_loc_idx)
    else:

        entity_start_idx = pre_tag.index(i_loc_idx)

    entity_list = []
    entity_list.append(sentence_list[entity_start_idx])
    for i in range(entity_start_idx + 1, pre_tag_len):
        if pre_tag[i] == i_loc_idx:
            entity_list.append(sentence_list[i])
        else:
            break
    return "".join(entity_list)


def semantic_matching(model, tokenizer, question, attribute_list, answer_list, max_length):
    assert len(attribute_list) == len(answer_list)

    pad_token = 0
    pad_token_segment_id = 1
    features = []
    for (ex_index, attribute) in enumerate(attribute_list):
        inputs = tokenizer.encode_plus(
            text=question,
            text_pair=attribute,
            add_special_tokens=True,
            max_length=max_length,
            truncate_first_sequence=True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)
        features.append(
            SimInputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    assert all_input_ids.shape == all_attention_mask.shape
    assert all_attention_mask.shape == all_token_type_ids.shape

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=128)

    data_num = all_attention_mask.shape[0]
    batch_size = 128

    all_logits = None
    for i in range(0, data_num, batch_size):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids': all_input_ids[i:i + batch_size].to(device),
                      'attention_mask': all_attention_mask[i:i + batch_size].to(device),
                      'token_type_ids': all_token_type_ids[i:i + batch_size].to(device),
                      'labels': None
                      }
            outputs = model(**inputs)
            logits = outputs[0]
            logits = logits.softmax(dim=-1)

            if all_logits is None:
                all_logits = logits.clone()
            else:
                all_logits = torch.cat([all_logits, logits], dim=0)
    pre_rest = all_logits.argmax(dim=-1)
    if 0 == pre_rest.sum():
        return torch.tensor(-1)
    else:
        return pre_rest.argmax(dim=-1)


# 文字直接匹配，看看属性的詞語在不在句子之中
def text_match(attribute_list, answer_list, sentence):
    assert len(attribute_list) == len(answer_list)

    idx = -1
    for i, attribute in enumerate(attribute_list):
        if attribute in sentence:
            idx = i
            break
    if -1 != idx:
        return attribute_list[idx], answer_list[idx]
    else:
        return "", ""


# 從clean_triple.csv中找出實體相關的屬性
def select_triple(entity_str):
    df_triple = pd.read_csv("./input/data/DB_Data/clean_triple.csv", encoding="utf-8")
    return df_triple.loc[df_triple['entity'] == entity_str]


def main():
    with torch.no_grad():
        tokenizer_inputs = ()
        tokenizer_kwards = {'do_lower_case': False,
                            'max_len': 128,
                            'vocab_file': './input/config/bert-base-chinese-vocab.txt'}
        ner_processor = NerProcessor()
        sim_processor = SimProcessor()
        tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)

        ner_model = get_ner_model(config_file='./input/config/bert-base-chinese-config.json',
                                  pre_train_model='./output/best_ner.bin', label_num=len(ner_processor.get_labels()))
        ner_model = ner_model.to(device)
        ner_model.eval()

        sim_model = get_sim_model(config_file='./input/config/bert-base-chinese-config.json',
                                  pre_train_model='./output/best_sim.bin',
                                  label_num=len(sim_processor.get_labels()))

        sim_model = sim_model.to(device)
        sim_model.eval()

        # 寫入答案
        fq = open("./input/data/NLPCC2016KBQA/Kbqa.testing-data", 'r', encoding='utf8')
        i = 1
        timestart = time.time()
        fo = open("./input/data/NLPCC2016KBQA/04_Kbqa.testing-data", 'w', encoding='utf8')
        fo.close()
        listq = []
        for line in fq:
            if line[1] == 'q':
                listq.append(line[line.index('\t') + 1:].strip())
        print("ListQ ready! Start to predict.....")
        for q in listq:
            fo = open("./input/data/NLPCC2016KBQA/04_Kbqa.testing-data", 'a', encoding='utf8')
            q = q.lower()
            fo.write('<question id=' + str(i) + '>\t' + q + '\n')
            entity = get_entity(model=ner_model, tokenizer=tokenizer, sentence=q, max_len=128)

            if len(entity) != 0:
                triple_list = select_triple(entity)

                if len(triple_list) > 0:
                    attribute_list = list(triple_list["attribute"])
                    answer_list = list(triple_list["answer"])

                    attribute, answer = text_match(attribute_list, answer_list, q)

                    if attribute != '' and answer != '':
                        ret = answer
                    else:
                        attribute_idx = semantic_matching(sim_model, tokenizer, q, attribute_list, answer_list, 128).item()

                        if -1 == attribute_idx:
                            ret = ''
                        else:
                            ret = answer_list[attribute_idx]
                else:
                    ret = ''
                # print("問題{}{}的答案是{}".format(i, q, ret))
                fo.write('<answer id=' + str(i) + '>\t')
                fo.write(str(ret))
                fo.write('\n==================================================\n')
            else:
                fo.write('<answer id=' + str(i) + '>\t')
                fo.write('\n==================================================\n')

            print('processing ' + str(i) + 'th Q.\tAv time cost: ' + str((time.time() - timestart) / i)[:6] + ' sec')
            fo.close()
            i += 1
        print('Finished prediction.')
        fq.close()


if __name__ == '__main__':
    main()
