import random
import torch
from mwptoolkit.utils.enum_type import SpecialTokens

class AbstractDataLoader(object):
    def __init__(self,config,dataset):
        super().__init__()
        self.device=config["device"]
        self.train_batch_size=config["train_batch_size"]
        self.test_batch_size=config["test_batch_size"]
        self.share_vocab=config["share_vocab"]
        self.equation_prefix=config["equation_fix"]
        self.symbol_for_tree=config["symbol_for_tree"]
        self.train_batch_size=config["train_batch_size"]
        self.test_batch_size=config["test_batch_size"]
        self.max_len=config["max_len"]
        self.max_equ_len=config["max_equ_len"]
        
        self.dataset=dataset
        self.in_pad_token=dataset.in_word2idx[SpecialTokens.PAD_TOKEN]
        self.in_unk_token=dataset.in_word2idx[SpecialTokens.UNK_TOKEN]
        
        if self.symbol_for_tree:
            self.out_pad_token=self.in_pad_token
            self.out_unk_token=dataset.out_symbol2idx[SpecialTokens.UNK_TOKEN]
            self.temp_unk_token=dataset.temp_symbol2idx[SpecialTokens.UNK_TOKEN]
        else:
            if self.share_vocab:
                self.out_pad_token=self.in_pad_token
                self.out_unk_token=self.in_unk_token
                self.temp_pad_token=self.in_pad_token
                self.temp_unk_token=self.in_unk_token
            else:
                self.out_pad_token=dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
                self.out_unk_token=dataset.out_symbol2idx[SpecialTokens.UNK_TOKEN]
                self.temp_pad_token=dataset.temp_symbol2idx[SpecialTokens.PAD_TOKEN]
                self.temp_unk_token=dataset.temp_symbol2idx[SpecialTokens.UNK_TOKEN]
                

    
    def _pad_input_batch(self,batch_seq,batch_seq_len):
        if self.max_len != None:
            max_length=self.max_len
        else:
            max_length=max(batch_seq_len)
        for idx,length in enumerate(batch_seq_len):
            if length<max_length:
                x=batch_seq[idx]+[self.in_pad_token for i in range(max_length-length)]
                batch_seq[idx]+=[self.in_pad_token for i in range(max_length-length)]
            else:
                batch_seq[idx]=batch_seq[idx][:max_length]
        return batch_seq
    
    def _pad_output_batch(self,batch_target,batch_target_len):
        if self.max_equ_len != None:
            max_length=self.max_equ_len
        else:
            max_length=max(batch_target_len)
        for idx,length in enumerate(batch_target_len):
            if length<max_length:
                batch_target[idx]+=[self.out_pad_token for i in range(max_length-length)]
            else:
                batch_target[idx]=batch_target[idx][:max_length]
        return batch_target
    
    def _get_mask(self,batch_seq_len):
        max_length=max(batch_seq_len)
        batch_mask=[]
        for idx,length in enumerate(batch_seq_len):
            batch_mask.append([1]*length+[0]*(max_length-length))
        return batch_mask
    
    def _build_num_stack(self, equation, num_list):
        num_stack = []
        for word in equation:
            temp_num = []
            flag_not = True
            if word not in self.dataset.out_idx2symbol:
                flag_not = False
                if "NUM" in word:
                    temp_num.append(int(word[4:]))
                for i, j in enumerate(num_list):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(num_list))])
        num_stack.reverse()
        return num_stack
    
    def load_data(self):
        raise NotImplementedError