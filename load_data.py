# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# datasets/slr18/slr18.py

"""
bostenai 定义的.THCHS-30数据库接口
数据集URL：http://openslr.org/18/
Identifier: SLR18
Summary: A Free Chinese Speech Corpus Released by CSLT@Tsinghua University
Category: Speech
License: Apache License v.2.0
@misc{THCHS30_2015,
  title={THCHS-30 : A Free Chinese Speech Corpus},
  author={Dong Wang, Xuewei Zhang, Zhiyong Zhang},
  year={2015},
  url={http://arxiv.org/abs/1512.01882}
}
"""

from __future__ import absolute_import, division, print_function
import os
import fnmatch
from multiprocessing import Pool
from functools import partial
import datasets

_CITATION = """\
@InProceedings{huggingface:dataset,
title = {THCHS-30},
author={bostenai, Inc.
},
year={2021}
}
"""

# 来自http://openslr.org/18/ 和我自己的理解
_DESCRIPTION = """\
THCHS30 is an open Chinese speech database published by Center for Speech and Language Technology (CSLT) at Tsinghua University. The origional recording was conducted in 2002 by Dong Wang, supervised by Prof. Xiaoyan Zhu, at the Key State Lab of Intelligence and System, Department of Computer Science, Tsinghua Universeity, and the original name was 'TCMSD', standing for 'Tsinghua Continuous Mandarin Speech Database'. The publication after 13 years has been initiated by Dr. Dong Wang and was supported by Prof. Xiaoyan Zhu. We hope to provide a toy database for new researchers in the field of speech recognition. Therefore, the database is totally free to academic users. You can cite the data using the following BibTeX entry:

@misc{THCHS30_2015,
  title={THCHS-30 : A Free Chinese Speech Corpus},
  author={Dong Wang, Xuewei Zhang, Zhiyong Zhang},
  year={2015},
  url={http://arxiv.org/abs/1512.01882}

本数据集封装支持本地数据源，设置环境变量 SLR18_Corpus 到解压后的数据集可以根目录
export $SLR18_Corpus=/path/to/slr18
解压后的SLR18目录的结构应该是：
data_thchs30 /
    data /
        *.wav
        *.trn
    train /
        *.wav
        *.trn
    dev
    test
    lm_phone
    lm_word
}
"""

# 指向openslr，如果需要加速请自行指向其他地址
_HOMEPAGE = "http://openslr.org/18/"

# 复制了thch30的license
_LICENSE = "Apache License v.2.0"

# 这里有三种配置，text返回文本/全拼音/声韵母分离
_URLs = {
    'thch30': "https://www.openslr.org/resources/18/data_thchs30.tgz",
    'pinyin1': "https://www.openslr.org/resources/18/data_thchs30.tgz",
    'pinyin2': "https://www.openslr.org/resources/18/data_thchs30.tgz",
}


# 主数据类
class Slr18Dataset(datasets.GeneratorBasedBuilder):
    """BostenAI 制作的thch30的数据集封装"""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="thch30", version=VERSION,
                               description="thch30的基础数据，speech data and transcripts，text对应中文"),
        datasets.BuilderConfig(name="pinyin1", version=VERSION, description="thch30的基础数据，text对应全量拼音，声调在后的模式"),
        datasets.BuilderConfig(name="pinyin2", version=VERSION, description="thch30的基础数据，text对应声韵母分离的拼音，声调在后的模式"),
    ]

    DEFAULT_CONFIG_NAME = "thch30"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "thch30":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),  #
                    "file": datasets.Value("string"),  # 音频文件
                    "pinyin1": datasets.Value("string"),  # 完整拼音
                    "pinyin2": datasets.Value("string"),  # 声韵母分离
                    "mandarin": datasets.Value("string"),  # 汉语文本
                    "text": datasets.Value("string"),  # 汉语文本
                }
            )
        elif self.config.name == "pinyin1":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),  #
                    "file": datasets.Value("string"),  # 音频文件
                    "pinyin1": datasets.Value("string"),  # 完整拼音
                    "pinyin2": datasets.Value("string"),  # 声韵母分离
                    "mandarin": datasets.Value("string"),  # 汉语文本
                    "text": datasets.Value("string"),  # 完整拼音
                }
            )
        elif self.config.name == "pinyin2":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),  #
                    "file": datasets.Value("string"),  # 音频文件
                    "pinyin1": datasets.Value("string"),  # 完整拼音
                    "pinyin2": datasets.Value("string"),  # 声韵母分离
                    "mandarin": datasets.Value("string"),  # 汉语文本
                    "text": datasets.Value("string"),  # 声韵母分离
                }
            )
        else:
            print("尚未实现的配置！")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        # 分析本地路径是否指定
        data_dir = os.getenv('SLR18_Corpus')
        use_local = False
        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            if os.path.exists(data_dir):
                use_local = True
        # 解压和分析数据
        if not use_local:
            my_urls = _URLs[self.config.name]
            data_dir = dl_manager.download_and_extract(my_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test"),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev"),
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(
            self, filepath, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        # 因为thch30数据集的train/dev/test的trn是指向上一级的
        assert os.path.exists(filepath), "数据集%s不存在" % (split)
        # 遍历读取数据
        ext = 'wav'
        file_paths = [os.path.join(dirpath, f)
                      for dirpath, dirnames, files in os.walk(filepath)
                      for f in fnmatch.filter(files, "*.%s" % (ext))]

        # 并发读取数据
        try:
            with Pool() as pool:
                res = pool.map(partial(slr18Corpus, split=split, config=self.config.name), file_paths)
                # 处理综合结果
            for i, iR in enumerate(res):
                if iR is None:
                    continue
                yield iR['id'], iR
        except Exception as e:
            raise Exception("执行并行逻辑异常！%s" % (e))


def slr18Corpus(item, split, config):
    """
    分析一个ITEM
    """
    tDir = os.path.dirname(item)
    tFname, text = os.path.splitext(item)
    tTrn = tFname + '.trn'
    try:
        if split in ['train', 'test', 'dev']:
            with open(tTrn, 'r') as f:
                line = f.readline()
            tTrn = os.path.realpath(os.path.join(tDir, line))
        # 读取trn文件
        with open(tTrn, 'r') as f:
            md = f.readline()  # 汉字
            p1 = f.readline()  # 全拼音
            p2 = f.readline()  # 声韵母分离
    except Exception:
        print("加载%s对应的trn文件失败！" % (item))
        return None
    key = os.path.splitext(os.path.split(item)[-1])[0]  # 文件名
    example = {
        "id": key,  #
        "file": item,  # 音频文件
        "pinyin1": p1,  # 完整拼音
        "pinyin2": p2,  # 声韵母分离
        "mandarin": md,  # 汉语文本
        'text': md
    }
    if config == 'pinyin1':
        example['text'] = p1
    elif config == 'pinyin2':
        example['text'] = p2

    return example
