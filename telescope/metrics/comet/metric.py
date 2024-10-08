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
from typing import Dict, List

import torch
from pytorch_lightning.trainer.trainer import Trainer
from telescope.metrics.comet.result import COMETResult
from telescope.metrics.metric import Metric
from torch.utils.data import DataLoader
from torch import tensor

from comet import download_model, load_from_checkpoint

if "COMET_MODEL" in os.environ:
    MODELNAME = os.environ["COMET_MODEL"]
else:
    MODELNAME = "Unbabel/wmt22-comet-da"


class COMET(Metric):

    name = "COMET"
    system_only = False

    def __init__(self, language=None, modelname: str = MODELNAME, **kwargs):
        self.modelname = modelname
        self.model = load_from_checkpoint(download_model(modelname))

    def prepare_sample(self, data: List[Dict[str, str]]) -> Dict[str, tensor]:
        return self.model.prepare_sample(data, stage="predict")

    def score(self, src: List[str], cand: List[str], ref: List[str], num_workers: int = 4) -> COMETResult:
        data = {"src": src, "mt": cand, "ref": ref}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        dataloader = DataLoader(
            dataset=data,
            batch_size=16,
            collate_fn=self.prepare_sample,
            num_workers=num_workers,
        )
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = [0] if torch.cuda.is_available() else "auto"
        trainer = Trainer(accelerator=accelerator, devices=devices, deterministic=False, logger=False)
        
        predictions = trainer.predict(
            self.model, dataloaders=dataloader, return_predictions=True
        )
        predictions = [i["scores"] for i in predictions]
        scores = torch.cat(predictions, dim=0).tolist()
        return COMETResult(
            sum(scores) / len(scores), scores, src, cand, ref, self.name, self.modelname
        )
