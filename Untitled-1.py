# %%
from transformers import AutoModelForCausalLM
from copy import deepcopy
import torch

model_id = 'facebook/opt-125m'
model = AutoModelForCausalLM.from_pretrained(model_id)
cp_model = deepcopy(model)

# %%
def l2_norm(x):
    return torch.linalg.matrix_norm(x)

# %%
import numpy as np
def QSNR(tensor1, tensor2):
    ratio = l2_norm(tensor1-tensor2) / l2_norm(tensor2)
    return -10*torch.log(ratio)
    

# %%
from quant import quant_unpack, dequant_unpack
bits = 16
gs = 16
scale, zero, qs = quant_unpack(bits, gs, cp_model)
q_x = dequant_unpack(scale, zero, qs, gs)
QSNRs = {}
for key in q_x.keys():
    if key.split('.')[-1] != 'lm_head':
        weight = key+'.weight'
        QSNRs[key] = round(QSNR(q_x[key], cp_model.state_dict()[weight]).item(), 3)

# %%
from quant import quant_unpack, dequant_unpack
bits = 16
gs = 32
scale, zero, qs = quant_unpack(bits, gs, cp_model)
q_x = dequant_unpack(scale, zero, qs, gs)

# %%
from quant import quant_unpack, dequant_unpack
bits = 16
gs = 16
scale, zero, qs = quant_unpack(bits, gs, cp_model)
q_x = dequant_unpack(scale, zero, qs, gs)

# %%
QSNRs

# %%



