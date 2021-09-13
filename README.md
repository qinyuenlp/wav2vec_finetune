<h2 align="center">
    <p>Fine-tune Wav2Vec2 with Transformers</p>
    <div>
        <img src="https://img.shields.io/badge/Python-3.6-green.svg"></img>
    	<img src="https://img.shields.io/badge/Pytorch-1.5.1-orange"></img>
		<img src="https://img.shields.io/badge/Transformers-3.4.0-blue"></img>
    </div>
</h2>

一份简易的Wav2Vec2微调代码。

### 1.依赖资源

#### 1.1 预训练模型

本项目是在[Wav2Vec2-Large-XLSR-53-Chinese-zh-CN](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn)的基础上进行微调，需要先自行下载该预训练模型到本地，存放到`pretrain/wav2vec2-large-xlsr-53-chinese-zh-cn`路径下：

```
.
├── pretrain
│   ├── README.md
│   ├── wav2vec-large-xlsr-53-chinese-zh-cn
│   │   ├── .gitattributes
│   │   ├── config.json
│   │   ├── flax_model.msgpack
│   │   ├── preprocessor_config.json
│   │   ├── pytorch_model.bin
│   │   ├── README.md
│   │   ├── special_tokens_map.json
│   │   └── vocab.json
```

#### 1.2 语音数据读取

由于在实践中可能存在训练数据与业务场景数据采样率不同的问题，本项目借用了[Z-zq](https://github.com/Z-yq)的[TensorflowASR项目中的语音数据读取代码](https://github.com/Z-yq/TensorflowASR/blob/master/utils/speech_featurizers.py)。  

### 2.数据集格式

训练数据分为**语音数据(.wav)**与**标注文本**，其中语音数据的采样率默认为16000。

- **注意：文本中最好不含英文字母，阿拉伯数字应转为汉字。**

二者存储在一个**.txt文件**中，格式如下:

```
{/path/to/wav_file1}\t{text1}
{/path/to/wav_file2}\t{text2}
...
```

例：

```
/home/username/aishell/BAC009S0002W0122\t而对楼市成交抑制作用最大的限购
/home/username/aishell/BAC009S0002W0123\t也成为地方政府的眼中钉
/home/username/aishell/BAC009S0002W0124\t自六月底呼和浩特市率先宣布取消限购后
```

训练数据以`train.txt`和`dev.txt`命名，放在`./data`路径下：

```
.
├── data
│   ├── dev.txt
│   ├── README.md
│   └── train.txt
```

### 3.模型训练与测试

#### 3.1 模型训练

`python train.py`

相关参数（epochs / batch_size / learning_rate / gpu or cpu）均可在`train.py`中设置。

#### 3.2 模型测试

先在`recognize.py`中设置`MODEL_DIR`与`wav_path`，然后运行：

`python test.py`