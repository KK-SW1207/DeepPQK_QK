import pickle
import os
import pickle
import os
import esm
import csv
import pandas as pd  # 导入pandas模块
import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import itertools
# from model3 import DeepDTAF
from model_split_poc import DeepDTAF
from datetime import datetime

device = torch.device("cuda")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
model = model.to(device)
# 获取当前时间
model2 = DeepDTAF()
# model2.load_state_dict(torch.load('best_model.pt'))
model2.load_state_dict(torch.load('best_model_split.pt'))
model2.eval()
model2 = model2.to(device)



CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)
PT_FEATURE_SIZE = 1280


def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] - 1

    return X


AA = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
# print(len(AA))
seq_mut = "MDVTTTIDQVLRNATETGEVPGVVAIAANDQEVIYEGAFGKRNINAAAPMTMDTIFRIASMTKAVTSVAAMQLVEQGKLQLDQPVASVIPAFGELQVLVGFYGDNPALRPPASQATIRHLLTHTSGLGYEIWNADLGRYQKVTNTPGIVSGQKAAFRNPLVADPGSTWNYGINTDWLGRVVEEVGGQTLGVYMRRNIFDPLGMKDTGFSETEEQKKRLVTVHARQADGSLAPIDFSWPAGREFENGGHGLVSTARDYLAFVRMLLNEGTYSGARVLRADTVAQMRQNHIGDLLVTMMKSANPAMSNDAEFFPGMKKKHGLGFVINTEQWPGMRAVGSCCWAGLFNSFYWFDPTKRIAAAIFMQILPFADPKAMEVYTAFEKAVYASV"
# 突变位点
# 口袋氨基酸位点
pkt_list = [60, 63, 129, 130, 131, 132, 149, 170, 237, 248, 297, 304, 310, 341, 343, 367]

smi = "CC(C)[C@@H]1C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O[C@@H](C(=O)N([C@H](C(=O)O1)CC2=CC=CC=C2)C)C(C)C)CC3=CC=CC=C3)C)C(C)C)CC4=CC=CC=C4)C"
smi_eb = label_smiles(smi, 150)
smi_eb = torch.from_numpy(smi_eb).to(device)
smi_eb = smi_eb.reshape(1, 150)
print(smi_eb.shape)
mut_seqlist = []

for i in range(1):

    mut_seq = seq_mut

    length = 1000
    mut_seq = mut_seq.ljust(length, "_")
    mut_seq = mut_seq[:length].rstrip()
    ling = "<pad>"
    mut_seq = mut_seq.replace("_", "<pad>")
    # print(mut_seq)
    data = [("DLFae4", mut_seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    token_representations = token_representations.cpu()
    token_representations = token_representations.numpy()
    token_representations = token_representations.reshape(-1, 1280)

    pocket_code = []
    for t in pkt_list:
        pocket_code.append(token_representations[t])
    pocket_code = np.array(pocket_code)
    # print(pocket_code.shape)

    seqlen_tensor = len(mut_seq)

    _seq_tensor = token_representations[1:seqlen_tensor - 1]
    _seq_tensor = _seq_tensor[0:1000]
    seq_tensor = np.zeros((1000, PT_FEATURE_SIZE))
    seq_tensor[0:len(_seq_tensor)] = _seq_tensor
    _pkt_tensor = pocket_code
    # 计算需要在第一维度上填充的数目
    pad1 = 63 - len(_pkt_tensor)
    if pad1 > 0:

        pkt_tensor = np.pad(_pkt_tensor, ((0, pad1), (0, 0)), 'constant', constant_values=(0, 0))
    else:

        pkt_tensor = _pkt_tensor[:63, :]
    seq_tensor = torch.from_numpy(seq_tensor).float().to(device)
    pkt_tensor = torch.from_numpy(pkt_tensor).float().to(device)
    seq_tensor = seq_tensor.reshape(1, 1000, 1280)
    pkt_tensor = pkt_tensor.reshape(1, 63, 1280)
    with torch.no_grad():
        predictions = model2(seq_tensor, pkt_tensor, smi_eb)

    predictions = np.array(predictions.cpu())
    predictions = str(predictions[0][0])
    print(predictions)


