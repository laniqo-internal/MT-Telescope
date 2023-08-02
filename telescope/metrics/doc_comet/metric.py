# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
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
import os
from typing import List

import torch
from pytorch_lightning.trainer.trainer import Trainer
from telescope.metrics.doc_comet.result import DocCOMETResult
from telescope.metrics.metric import Metric
from telescope.metrics.doc_comet.download_utils import download_model
from telescope.metrics.doc_comet.models import load_from_checkpoint
from torch.utils.data import DataLoader
# from comet import download_model, load_from_checkpoint

from .add_context import add_context


class DocCOMET(Metric):

    name = "DocCOMET"
    system_only = False

    def __init__(self, language=None, modelname: str = "Unbabel/wmt22-comet-da", **kwargs):
        self.modelname = modelname
        self.model = load_from_checkpoint(download_model(modelname))
        self.model.set_document_level()

    def prepare_sample(self, data):
        return self.model.prepare_sample(data, inference=True)
    
    def score(self, src: List[str], cand: List[str], ref: List[str], doc_ids: List[str]) -> DocCOMETResult:
        src = add_context(orig_txt=src, context=src, doc_ids=doc_ids, sep_token=self.model.encoder.tokenizer.sep_token)
        cand = add_context(orig_txt=cand, context=ref, doc_ids=doc_ids, sep_token=self.model.encoder.tokenizer.sep_token)
        ref = add_context(orig_txt=ref, context=ref, doc_ids=doc_ids, sep_token=self.model.encoder.tokenizer.sep_token)
        data = {"src": src, "mt": cand, "ref": ref}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        cuda = 1 if torch.cuda.is_available() else 0
        seg_scores, scores = self.model.predict(data, batch_size=16, gpus=cuda)
        return DocCOMETResult(
            sum(seg_scores) / len(seg_scores), scores, src, cand, ref, self.name, self.modelname
        )
