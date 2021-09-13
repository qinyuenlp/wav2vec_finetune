from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch
from speech_featurizers import speech_config, SpeechFeaturizer
import os

MODEL_DIR = "./saved_model/20210910"

model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR)
processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)

# wav_path = 'E:/DATASETS/ASR/12368/chongqing/left_real/2021-01-05-14-10-16_3050_15204440078.wav'  # 判决书
# wav_path = 'D:/Codes/Aegis/ASR/DATA/12368_iflytek/test/chongqing/ffd3e0ce-06dd-11ec-89c4-8dbbecec51f6.wav'  # 江口九龙坡法院
# wav_path = 'D:/Codes/Aegis/ASR/DATA/12368_iflytek/test/chongqing/1ad26b98-06de-11ec-89c4-8dbbecec51f6.wav'  # 观音桥
# wav_path = 'D:/Codes/Aegis/ASR/DATA/12368_iflytek/test/chongqing/fd67740e-06dd-11ec-89c4-8dbbecec51f6.wav'  # 找我律师是吧
# wav_path = 'D:/Codes/Aegis/ASR/DATA/12368_iflytek/test/chongqing/2021-01-04-13-50-25_3050_18716868804.wav'  # 重庆市江北法院
# wav_path = 'D:/Codes/Aegis/ASR/DATA/那您现在的案件处理到哪一步了.wav'
# wav_path = 'D:/Codes/Aegis/ASR/DATA/008d622c-12c0-11ec-b9cd-51692d7c80d8.wav'
# wav_path = 'D:/Codes/Aegis/ASR/DATA/4d6d724a-12bf-11ec-b9cd-51692d7c80d8.wav'
wav_path = 'D:/Codes/Aegis/ASR/DATA/业务测试/asr测试样例4.wav'

# wav, sr = sf.read(wav_path)

sp = SpeechFeaturizer(speech_config)

for filename in os.listdir('../DATA/业务测试'):
    wav_path = f"../DATA/业务测试/{filename}"
    wav = sp.load_wav(wav_path)

    inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)[0]
    print(f"{filename} : {predicted_sentences}")
    a = 1