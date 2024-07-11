import os
import re
from copy import deepcopy
from warnings import filterwarnings
import nltk
import torch

from mwptoolkit.config import Config
from mwptoolkit.data.utils import get_dataset_module, get_dataloader_module
from mwptoolkit.utils.enum_type import SpecialTokens, NumMask
from mwptoolkit.utils.preprocess_tool.number_operator import english_word_2_num
from mwptoolkit.utils.preprocess_tool.number_transfer import get_num_pos
from mwptoolkit.utils.utils import get_model, str2float

filterwarnings('ignore')

def main():
    # a MWP sample
    problem = "3*10"
    trained_model_dir = "./trained_model/Saligned-mawps-single"
    # load config
    config = Config.load_from_pretrained(trained_model_dir)
    # load dataset parameters
    dataset = get_dataset_module(config).load_from_pretrained(config['trained_model_dir'])
    dataset.dataset_load()
    dataloader = get_dataloader_module(config)(config, dataset)
    # load model parameters
    model = get_model(config['model'])(config, dataset)
    model_file = os.path.join(config['trained_model_dir'], 'model.pth')
    state_dict = torch.load(model_file, map_location=config["map_location"])
    model.load_state_dict(state_dict["model"], strict=False)

    # preprocess
    language = 'english' if config['language'] == 'en' else 'zh'
    word_list = nltk.word_tokenize(problem, language)
    word_list = english_word_2_num(word_list,fraction_acc=2)
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?|(-\d+)")

    # input_seq, num_list, final_pos, all_pos, nums, num_pos_dict, nums_for_ques, nums_fraction
    process_data = get_num_pos(word_list, config['mask_symbol'], pattern)
    source = deepcopy(process_data[0])
    for pos in process_data[3]:
        for key, value in process_data[5].items():
            if pos in value:
                num_str = key
                break
        num = str(str2float(num_str))
        source[pos] = num
    source = ' '.join(source)
    data_dict = {'question': process_data[0], 'number list': process_data[1], 'number position': process_data[2],
                 'ques source 1': source}

    # build batch data
    batch = dataloader.build_batch_for_predict([data_dict])

    # predict
    token_logits,symbol_outputs,_ = model.to('cuda').predict(batch)

    # output process
    symbol_list = dataloader.convert_idx_2_symbol(symbol_outputs[0])
    equation = []
    for symbol in symbol_list:
        if symbol not in [SpecialTokens.SOS_TOKEN, SpecialTokens.EOS_TOKEN, SpecialTokens.PAD_TOKEN]:
            equation.append(symbol)
        else:
            break

    def trans_symbol_2_number(equ_list, num_list):
        symbol_list = NumMask.number
        new_equ_list = []
        for symbol in equ_list:
            if 'NUM' in symbol:
                index = symbol_list.index(symbol)
                if index >= len(num_list):
                    new_equ_list.append(symbol)
                else:
                    new_equ_list.append(str(num_list[index]))
            else:
                new_equ_list.append(symbol)
        return new_equ_list

    equation = trans_symbol_2_number(equation, data_dict['number list'])
    # final equation
    result = eval(''.join(equation))
    print(''.join(equation),"=",result)


if __name__ == '__main__':
    main()
