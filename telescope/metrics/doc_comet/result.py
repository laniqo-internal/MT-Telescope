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
from telescope.metrics.result import MetricResult


class DocCOMETResult(MetricResult):
    def __init__(
        self,
        sys_score: int,
        seg_scores: list,
        src: list,
        cand: list,
        ref: list,
        metric: str,
        model: str,
    ) -> None:
        super().__init__(sys_score, seg_scores, src, cand, ref, metric)
        self.model = model

    def __str__(self):
        return f"{self.metric}({self.sys_score}, Model={self.model})"
