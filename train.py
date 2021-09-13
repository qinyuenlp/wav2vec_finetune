from torch.nn import CTCLoss
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup, WEIGHTS_NAME, CONFIG_NAME
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import torch
from tqdm import tqdm
import numpy as np
from jiwer import wer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

def default_loader(path):
    wav, sample_rate = sf.read(path)
    return torch.Tensor(wav)

def collate_fn(batch_data, pad=0.):
    max_len = max([len(seq) for seq in batch_data])
    for i in range(len(batch_data)):
        wav = batch_data[i]
        wav += [pad]*(max_len-len(wav))
    return batch_data

class ASRDataset(Dataset):
    def __init__(self, index_file, loader=default_loader):
        # 定义好 image 的路径
        self.wavs = list()
        self.target = list()
        self._load_index(index_file)
        self.loader = loader

    def _load_index(self, file):
        wavs, target = [], []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                wav_file, text = line[:-1].split('\t')
                wavs.append(wav_file)
                target.append(text)
        self.wavs = wavs
        self.target = target

    def loader(self, file):
        wav, _ = sf.read(file, samplerate=16000)
        return wav.tolist()

    def __getitem__(self, index):
        wav = self.wavs[index]
        target = self.target[index]
        return wav, target

    def __len__(self):
        return len(self.wavs)


def evaluate(model, loss_fn, tokenizer, processor, dev_data, DEVICE):
    model.eval()
    accuracy = []
    total_loss = 0
    with torch.no_grad():
        for file, real in dev_data:
            wav, _ = sf.read(file)
            inputs = processor(wav, sampling_rate=16_000, return_tensors="pt")
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

            log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
            targets, target_lengths = [], []
            targets.extend(tokenizer(real).data['input_ids'])
            target_lengths.append(len(real))

            input_lengths = torch.full(size=(log_probs.shape[1],), fill_value=log_probs.shape[0], dtype=torch.long).to(DEVICE)

            targets = torch.Tensor(targets).to(DEVICE)
            target_lengths = torch.IntTensor(target_lengths).to(DEVICE)

            loss = loss_fn(log_probs=log_probs, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths)
            total_loss += loss

            predicted_ids = torch.argmax(logits, dim=-1).cpu().detach()
            predicted_sentences = processor.batch_decode(predicted_ids)[0]

            pred = ' '.join(list(predicted_sentences.replace(' ', '').replace('<unk>', '').replace('<pad>', '').replace('<s>', '').replace('</s>', '')))
            real = ' '.join(list(real))
            file_wer = wer(truth=real, hypothesis=pred)
            accuracy.append(file_wer)
    return loss, 1 - np.mean(accuracy)

def save_model(save_dir, model, processor):
    model_to_save = model.module if hasattr(model, "module") else model

    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(save_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)

    model_to_save.config.to_json_file(output_config_file)
    processor.save_pretrained(save_dir)


if __name__ == "__main__":

    epochs = 30
    batch_size = 32
    learning_rate = 3e-5

    TRAIN_DATA_PATH = "./data/train.txt"
    DEV_DATA_PATH = "./data/dev.txt"
    MODEL_DIR = "./pretrain/wav2vec2-large-xlsr-53-chinese-zh-cn"  # 预训练模型目录路径
    SAVE_DIR = "./saved_model"  # 模型保存路径

    if os.environ["CUDA_VISIBLE_DEVICES"] == '-1':
        DEVICE = torch.device('cpu')
    else:
        DIVICE = torch.device('cuda')

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_DIR)
    config = Wav2Vec2Config.from_pretrained(MODEL_DIR)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)

    ## Load Dataset
    train_data = ASRDataset(TRAIN_DATA_PATH)
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    dev_data = [i.split('\t') for i in open(DEV_DATA_PATH, 'r', encoding='utf-8')]

    model = model.to(DEVICE)
    model.train()


    ## TRAINING
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05, num_training_steps=len(train_iter)*epochs)
    loss_fn = CTCLoss(blank=0, zero_infinity=False)

    total_batch = 0
    best_dev_acc = -float('inf')
    for epoch in range(epochs):
        for batch in tqdm(train_iter, total=len(train_iter)//batch_size, desc=f"epoch-{epoch}"):
            files, texts = batch
            wavs = [sf.read(path)[0].tolist() for path in files]

            inputs = processor(wavs, sampling_rate=16_000, return_tensors="pt", padding=True)
            logits = model(inputs.input_values.to(DEVICE), attention_mask=inputs.attention_mask.to(DEVICE)).logits
            model.zero_grad()

            log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
            targets, target_lengths = [], []
            for text in texts:
                targets.extend(tokenizer(text).data['input_ids'])
                target_lengths.append(len(text))

            input_lengths = torch.full(size=(log_probs.shape[1],), fill_value=log_probs.shape[0], dtype=torch.long).to(DEVICE)

            targets = torch.Tensor(targets).to(DEVICE)
            target_lengths = torch.IntTensor(target_lengths).to(DEVICE)

            loss = loss_fn(log_probs=log_probs, targets=targets,
                           input_lengths=input_lengths, target_lengths=target_lengths)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if total_batch % 20 == 0:
                dev_loss, dev_acc = evaluate(model, loss_fn, tokenizer, processor, dev_data, DEVICE)
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    save_model(SAVE_DIR, model, processor)
                    improve = '*'
                else:
                    improve = ' '
                print(f"\n{improve} Loss of Dev = {dev_loss}, Accuracy of Dev = {dev_acc}\n")

            optimizer.step()

            total_batch += 1