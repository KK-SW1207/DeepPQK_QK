from pathlib import Path
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from typing import List

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
    X = np.zeros(max_smi_len, dtype=np.int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] - 1

    return X


class MyDataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len, max_pkt_len, max_smi_len, pkt_window=None, pkt_stride=None):
        data_path = Path(data_path)

        affinity = {}
        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity

        ligands_df = pd.read_csv(data_path / f"{phase}_smi.csv")
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        
        seqlen_df = pd.read_csv(data_path / f"{phase}_seq_.csv")
        seqlen = {i["id"]: i["seq"] for _, i in seqlen_df.iterrows()}
        self.seqlen = seqlen
        
        #prefile="../pre_data/{}/global".format(phase)
        prefile="../DeepDTAF-master/pre_data/{}/global".format(phase)
        mfile_list = os.listdir(prefile)
        nfile_list=[]

        for m in mfile_list:
            a=m.split(".")[0]
            nfile_list.append(a)
        
        ligands= {k: v for k, v in ligands.items() if k in  nfile_list}  
        self.smi = ligands
        self.max_smi_len = max_smi_len

        seq_path = data_path / phase / 'global'
        pkt_path =data_path / phase / 'pocket'
        self.seq_path = sorted(list(seq_path.glob('*.npy')))
        #print(self.seq_path)
        self.max_seq_len = max_seq_len
        self.pkt_path = sorted(list(pkt_path.glob('*.npy')))
        print(len(sorted(list(pkt_path.glob('*.npy')))))
        self.max_pkt_len = max_pkt_len
        self.pkt_window = pkt_window
        self.pkt_stride = pkt_stride
        if self.pkt_window is None or self.pkt_stride is None:
            print(f'Dataset {phase}: will not fold pkt')

        assert len(self.seq_path) == len(self.pkt_path)
        assert len(self.seq_path) == len(self.smi)
        
        self.length = len(self.smi)

    def __getitem__(self, idx):

        seq = self.seq_path[idx]
        pkt = self.pkt_path[idx]

        assert seq.name == pkt.name
        seqlen_tensor=len(self.seqlen[seq.name.split('.')[0]])
        
        _seq_tensor=np.load(seq, allow_pickle=True)[1:seqlen_tensor-1]
        _seq_tensor=_seq_tensor[0:1000]
        seq_tensor = np.zeros((self.max_seq_len, PT_FEATURE_SIZE))
        seq_tensor[0:len(_seq_tensor)] = _seq_tensor
        _pkt_tensor=np.load(pkt, allow_pickle=True)

        pad1 = 63 - len(_pkt_tensor)
        if pad1 > 0:

            pkt_tensor = np.pad(_pkt_tensor,((0,pad1),(0,0)),'constant',constant_values = (0,0))
        else:

            pkt_tensor = _pkt_tensor[:63, :]


        return (seq_tensor.astype(np.float32),
                pkt_tensor.astype(np.float32),
                label_smiles(self.smi[seq.name.split('.')[0]], self.max_smi_len),
                np.array(self.affinity[seq.name.split('.')[0]], dtype=np.float32))

    def __len__(self):
        return self.length
    