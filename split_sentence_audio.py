import os
import logging
import queue
import json
import numpy as np
from scipy.io import wavfile
import librosa
import soundfile as sf

import webrtcvad
vad = webrtcvad.Vad(3)
sample_rate = 16000

def split_audio_file(file_path):
    sample_rate,wav_data = wavfile.read(file_path)
    left_data = []
    for item in wav_data:
        left_data.append(item[0])
    file_name = os.path.basename(file_path)
    wavfile.write("left/"+file_name, sample_rate, np.array(left_data))

    new_data = []
    sample_rate,wav_data = wavfile.read("left/"+file_name)

    begin = False
    end = False

    for i in range(0, len(wav_data), 320):
        flag = vad.is_speech(wav_data[i:i+320].tobytes(), 16000)
        if flag:
            begin = True
            end = False
            new_data.extend(wav_data[i:i+320])
        else:
            if not end and begin:
                if new_data and begin:
                    if len(new_data) < 30*1024:
                        end = True
                        begin = False
                        new_data = None
                        new_data = []
                        continue
                    import uuid
                    _file_name = str(uuid.uuid1()) + ".wav"
                    wavfile.write("left_split_real/"+_file_name, 8000, np.array(new_data))
                    src_sig,sr = sf.read("left_split_real/"+_file_name)  #name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率  
                    dst_sig = librosa.resample(src_sig,sr,16000)  #resample 入参三个 音频数据 原采样频率 和目标采样频率
                    sf.write("left_split_real/"+_file_name,dst_sig,16000) #写出数据  参数三个 ：  目标地址  更改后的音频数据  目标采样数据

                    end = True
                    begin = False
                    new_data = None
                    new_data = []



if __name__ == "__main__":
    file_list = os.listdir("2021_04")
    for file_path in file_list:
        file_final_path = "2021_04/" + file_path
        try:
            split_audio_file(file_final_path)
        except Exception as e:
            print(e)
            continue
        # sample_rate, data = wavfile.read(file_final_path)
        # in_queue = queue.Queue(maxsize=5000)
        # out_queue = queue.Queue(maxsize=5000)
        # from asr_client import ThirdSocket
        # in_queue.put(file_final_path)
        # ThirdSocket().do(in_queue, out_queue)
        # f_text = ""
        # while not out_queue.empty():
        #     text = out_queue.get()
        #     f_text += text
        # print(file_final_path + "\t" + f_text)
        #     #print("text: " + text)


