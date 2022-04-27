# coding=utf-8
# Copyright 2022 rinna Co., Ltd.
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

import torch
import torch.nn.functional as F


def cloob_loss(image_features, text_features, inv_tau, scale_hopfield):
    """
    Note: this loss has been rescaled from the original CLOOB loss for interpretability,
    to convert to the original, divide it by inv_tau / 2.
    """
    p_xx, p_yy, p_xy, p_yx = hopfield_retrieval(image_features, text_features, scale_hopfield)
    identity = torch.eye(p_xx.shape[1]) > 0.5
    i = identity.to(p_xx.device)
    loss_img = infoLOOB_loss(p_xx.T, p_xy.T, i, inv_tau=inv_tau)
    loss_txt = infoLOOB_loss(p_yy.T, p_yx.T, i, inv_tau=inv_tau)
    # return loss_img + loss_txt
    # https://github.com/crowsonkb/cloob-training/blob/master/cloob_training/loss.py#L27
    return (loss_img + loss_txt) / 2


def infoLOOB_loss(x, y, i, inv_tau):
    tau = 1 / inv_tau
    k = x @ y.T / tau
    positives = -torch.mean(torch.sum(k * i, dim=1))

    # For logsumexp the zero entries must be equal to a very large negative number
    large_neg = -10000.0
    arg_lse = k * torch.logical_not(i) + i * large_neg
    negatives = torch.mean(torch.logsumexp(arg_lse, dim=1))
    # crowsonkb's implementation
    # https://github.com/crowsonkb/cloob-training/blob/master/cloob_training/loss.py#L27
    return positives + negatives
    # return tau * (positives + negatives)


def hopfield_retrieval(image_features, text_features, scale_hopfield):
    patterns_xx = hopfield(state_patterns=image_features, stored_patterns=image_features, scale_hopfield=scale_hopfield)
    patterns_yy = hopfield(state_patterns=text_features, stored_patterns=text_features, scale_hopfield=scale_hopfield)
    patterns_xy = hopfield(state_patterns=text_features, stored_patterns=image_features, scale_hopfield=scale_hopfield)
    patterns_yx = hopfield(state_patterns=image_features, stored_patterns=text_features, scale_hopfield=scale_hopfield)

    return patterns_xx, patterns_yy, patterns_xy, patterns_yx


def hopfield(state_patterns, stored_patterns, scale_hopfield):
    # retrieved_patterns = hopfield_layer.forward(
    #     (stored_patterns.unsqueeze(0), state_patterns.unsqueeze(0), stored_patterns.unsqueeze(0))).squeeze()
    # crowsonkb's implementation
    # https://github.com/crowsonkb/cloob-training/blob/master/cloob_training/loss.py#L38
    retrieved_patterns = stored_patterns.T @ F.softmax(scale_hopfield * stored_patterns @ state_patterns.T, dim=0)
    # Row vectors -> dim=1 to normalize the row vectors
    retrieved_patterns = retrieved_patterns / retrieved_patterns.norm(dim=0, keepdim=True)
    return retrieved_patterns
