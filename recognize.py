from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class W2V(object):
    def __init__(self, model_dir, device='cpu'):
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir)
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
        self.device = device
        self.model.to(self.device)

    def recognize(self, wav_path):
        wav, _ = sf.read(wav_path)

        inputs = self.processor(wav, sampling_rate=16_000, return_tensors="pt")
        logits = self.model(inputs.input_values.to(self.device), attention_mask=inputs.attention_mask.to(self.device)).logits

        predicted_ids = torch.argmax(logits, dim=-1).cpu().detach()
        predicted_sentences = self.processor.batch_decode(predicted_ids)[0].replace(' ', '').replace('<unk>', '').replace('<pad>', '').replace('<s>', '').replace('</s>', '')

        return predicted_sentences


if __name__ == "__main__":
    MODEL_DIR = "./saved_model"
    asr = W2V(MODEL_DIR)

    wav_path = '/your/path/to/test.wav'
    res = asr.recognize(wav_path)
    print(res)