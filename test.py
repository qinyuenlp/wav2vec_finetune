from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from speech_featurizers import speech_config, SpeechFeaturizer

MODEL_DIR = "./saved_model"

model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)

sp = SpeechFeaturizer(speech_config)

wav_path = 'test.wav'

wav = sp.load_wav(wav_path)

inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)[0]
print(predicted_sentences)