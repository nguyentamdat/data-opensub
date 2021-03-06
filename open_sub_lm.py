# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
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

# Lint as: python3
import os
import re
import datasets


_DESCRIPTION = """\
This is a new collection of translated movie subtitles from http://www.opensubtitles.org/.

IMPORTANT: If you use the OpenSubtitle corpus: Please, add a link to http://www.opensubtitles.org/ to your website and to your reports and publications produced with the data!

This is a slightly cleaner version of the subtitle collection using improved sentence alignment and better language checking.

62 languages, 1,782 bitexts
total number of files: 3,735,070
total number of tokens: 22.10G
total number of sentence fragments: 3.35G
"""
_HOMEPAGE_URL = "http://opus.nlpl.eu/OpenSubtitles.php"
_CITATION = """\
P. Lison and J. Tiedemann, 2016, OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)
"""

_VERSION = "2018.0.0"
_BASE_NAME = "{}.txt"
_BASE_URL = "http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.{}.gz"

# Please note that only few pairs are shown here. You can use config to generate data for all language pairs
_LANGUAGE = ["vi"]


class OpenSubtitlesConfig(datasets.BuilderConfig):
    def __init__(self, *args, lang=None, min_len=9, max_len=128, max_context=10, eos="<EOS>", bos="<BOS>", pad="<PAD>", **kwargs):
        super().__init__(
            *args,
            name=f"{lang}",
            **kwargs,
        )
        self.lang = lang
        self.min_len = min_len
        self.max_len = max_len
        self.max_context = max_context
        self.eos_token = eos
        self.bos_token = bos
        self.pad_token = pad


class OpenSubtitles(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        OpenSubtitlesConfig(
            lang=lang,
            description=f"Corpus {lang}",
            version=datasets.Version(_VERSION),
        )
        for lang in _LANGUAGE
    ]
    BUILDER_CONFIG_CLASS = OpenSubtitlesConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "doc": datasets.Value("string")
                },
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        def _base_url(lang):
            return _BASE_URL.format(lang)

        download_url = _base_url(self.config.lang)
        path = dl_manager.download_and_extract(download_url)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"datapath": path},
            )
        ]

    def _preprocess_line(self, line):
        # line = line.decode("utf-8")

        # Remove the first word if it is followed by colon (speaker names)
        # NOTE: this wont work if the speaker's name has more than one word
        line = re.sub('(?:^|(?:[.!?]\\s))(\\w+):', "", line)

        # Remove anything between brackets (corresponds to acoustic events).
        line = re.sub("[\\[(](.*?)[\\])]", "", line)

        # Strip blanks hyphens and line breaks
        line = line.strip(" -\n!")

        return line

    def _should_skip(self, line):
        """Whether a line should be skipped depending on the length."""
        numbers = sum(c.isdigit() for c in line)
        return len(line) < self.config.min_len or len(line) > self.config.max_len or numbers > len(line) / 2

    def _generate_examples(self, datapath):
        with open(datapath, encoding="utf-8") as f1:
            sentence_counter = 0
            for x in f1:
                x = self._preprocess_line(x.strip())
                if (self._should_skip(x)):
                    continue
                x = self.config.bos_token + x + self.config.eos_token

                result = (
                    sentence_counter,
                    {
                        "doc": x
                    },
                )
                sentence_counter += 1
                yield result
