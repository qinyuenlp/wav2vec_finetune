import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch.nn import CTCLoss

LANG_ID = "zh-CN"
MODEL_ID = "D:/Codes/PRETRAIN/ASR/wav2vec2-large-xlsr-53-chinese-zh-cn"
SAMPLES = 10

# test_dataset = load_dataset("common_voice", LANG_ID, split=f"test[:{SAMPLES}]")
# test_dataset = load_dataset("common_voice", LANG_ID, split=f"test[:{SAMPLES}]")
test_dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["file"], sr=16_000)
    batch["speech"] = speech_array
    batch["text"] = batch["text"].upper()
    return batch

loss_fn = CTCLoss()

test_dataset = test_dataset.map(speech_file_to_array_fn)
t1 = test_dataset["speech"][:2]
for i, each in enumerate([t1]):
    inputs = processor(each, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)

    # for i, predicted_sentence in enumerate(predicted_sentences):

    print("-" * 100)
    print("Reference:", test_dataset[:2]["text"])
    print("Prediction:", predicted_sentences)
    a = 1

# from asrecognition import ASREngine
#
# asr = ASREngine("zh-CN", model_path="D:/Codes/PRETRAIN/ASR/wav2vec2-large-xlsr-53-chinese-zh-cn")
#
# print('init model')
# # audio_paths = ["E:/DATASETS/ASR/12368/chongqing/left_real/2021-01-05-14-10-16_3050_15204440078.wav"]
# # audio_paths = ["E:/DATASETS/ASR/12368/chongqing/left_real/2021-01-05-14-10-16_3050_15204440078.wav"]
# # audio_paths = ["D:/Codes/Aegis/ASR/DATA/12368_iflytek/chongqing/ffd3e0ce-06dd-11ec-89c4-8dbbecec51f6.wav"]
# audio_paths = ["D:/Codes/Aegis/ASR/DATA/12368_iflytek/chongqing/1ad26b98-06de-11ec-89c4-8dbbecec51f6.wav"]
#
# res = asr.transcribe(audio_paths)
# print(res)
a = 1