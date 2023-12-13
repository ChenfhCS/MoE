import time
import pickle
import numpy as np
import nltk
import evaluate
import collections
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import torch
# Load model directly
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DefaultDataCollator,
                            SwitchTransformersForConditionalGeneration,
                            AutoConfig, get_scheduler, DataCollatorForSeq2Seq,
                            )
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
# import evaluate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def save_model(model,name):
    save_path = dir_path + '/pth/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_to_save = model.module if hasattr(model, 'module') else model
    # model_checkpoint = os.path.join(dir_path + '/pth/', "%s_checkpoint.bin" % name)
    torch.save(model_to_save.state_dict(), save_path+f'moe_{name}_checkpoint.bin')
    print(f"Saved model checkpoint to {save_path}!")

def Create_MoE_Model(**kwargs):
    # transformer-xl
    if kwargs['model_name'] == 'xl':
        from transformers import TransfoXLForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
        config_load = AutoConfig.from_pretrained("transfo-xl-wt103")
        config = AutoConfig.from_pretrained("transfo-xl-wt103")
        config.moe = kwargs['moe']
        config.moe_num_experts = kwargs['moe_num_experts']
        config.moe_top_k = kwargs['moe_top_k']
        config.moe_group = kwargs['moe_group']
        config.moe_world_size = kwargs['moe_world_size']
        # config.num_labels = 2
        modelForLoad = TransfoXLForSequenceClassification(config=config_load)
        if config.moe_num_experts == 0:
            return modelForLoad,tokenizer
        # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        # outputs = modelForLoad(**inputs)
        mymoe = TransfoXLForSequenceClassification(config=config)
        # print(modelForLoad.state_dict().keys(),mymoe.state_dict().keys())
        mymoeParam = mymoe.state_dict()
        bertParam = modelForLoad.state_dict()
        # original weight = ['transformer.layers.0.pos_ff.CoreNet.0.weight', 'transformer.layers.0.pos_ff.CoreNet.0.bias', 
        # 'transformer.layers.0.pos_ff.CoreNet.3.weight', 
        # 'transformer.layers.0.pos_ff.CoreNet.3.bias']
        # desity weight = ['transformer.h.11.moe_linear.experts.15.htoh4.weight', 'transformer.h.11.moe_linear.experts.15.htoh4.bias', 
        # 'transformer.h.11.moe_linear.experts.15.h4toh.weight', 'transformer.h.11.moe_linear.experts.15.h4toh.bias',]
        # original_layer_normal = ['transformer.layers.0.pos_ff.layer_norm.weight', 'transformer.layers.0.pos_ff.layer_norm.bias']
        # desitny weight = ['transformer.h.11.moe_linear.layer_norm.weight', 'transformer.h.11.moe_linear.layer_norm.bias',]

        # copy linear weight, bias and layernormal
        for layer in range(config.n_layer):
            for expert_id in range(config.moe_num_experts):
                mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.weight'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.CoreNet.0.weight'].unsqueeze(0).detach().clone()
                mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.bias'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.CoreNet.0.bias'].unsqueeze(0).detach().clone()
                mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.weight'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.CoreNet.3.weight'].unsqueeze(0).detach().clone()
                mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.bias'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.CoreNet.3.bias'].unsqueeze(0).detach().clone()
            mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.layer_norm.weight'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.layer_norm.weight'].detach().clone()
            mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.layer_norm.bias'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.layer_norm.bias'].detach().clone() 
        mymoe.load_state_dict(mymoeParam)
        return mymoe, tokenizer

    # bert
    elif kwargs['model_name'] == 'bert':
        from transformers import BertForQuestionAnswering
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        config = AutoConfig.from_pretrained("bert-large-uncased")
        config_load = AutoConfig.from_pretrained("bert-large-uncased")
        config.moe = kwargs['moe']
        config.moe_num_experts = kwargs['moe_num_experts']
        config.moe_top_k = kwargs['moe_top_k']
        config.moe_group = kwargs['moe_group']
        config.moe_world_size = kwargs['moe_world_size']

        modelForLoad = BertForQuestionAnswering(config=config_load)
        if config.moe_num_experts == 0:
            return modelForLoad,tokenizer

        mymoe = BertForQuestionAnswering(config=config)
        # print(modelForLoad.state_dict().keys(),mymoe.state_dict().keys())
        mymoeParam = mymoe.state_dict()
        bertParam = modelForLoad.state_dict()
        # original weight = ['bert.encoder.layer.10.intermediate.dense.weight', 'bert.encoder.layer.10.intermediate.dense.bias', 
        # 'bert.encoder.layer.10.output.dense.weight', 'bert.encoder.layer.10.output.dense.bias']
        # desity weight = ['bert.encoder.layer.0.moe_linear.experts.0.htoh4.weight', 'bert.encoder.layer.0.moe_linear.experts.0.htoh4.bias', 
        # 'bert.encoder.layer.0.moe_linear.experts.0.h4toh.weight', 'bert.encoder.layer.0.moe_linear.experts.0.h4toh.bias',]
        # original_layer_normal = ['bert.encoder.layer.11.output.LayerNorm.weight', 'bert.encoder.layer.11.output.LayerNorm.bias']
        # desitny weight = ['bert.encoder.layer.0.moe_linear.layer_norm.weight', 'bert.encoder.layer.0.moe_linear.layer_norm.bias']
        # bertLayerLength=24
        # copy linear weight, bias and layernormal
        for layer in range(config_load.num_hidden_layers):
            for expert_id in range(config.moe_num_experts):
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.intermediate.dense.weight'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.intermediate.dense.bias'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.dense.weight'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.dense.bias'].unsqueeze(0).detach().clone()
            mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.layer_norm.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.LayerNorm.weight'].detach().clone()
            mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.layer_norm.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.LayerNorm.bias'].detach().clone() 
        mymoe.load_state_dict(mymoeParam)
        return mymoe, tokenizer
    
    # gpt2
    elif kwargs['model_name'] == 'gpt':
        from transformers import GPT2LMHeadModel, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = AutoConfig.from_pretrained("gpt2")
        config_load = AutoConfig.from_pretrained("gpt2")
        config.moe = kwargs['moe']
        config.moe_num_experts = kwargs['moe_num_experts']
        config.moe_top_k = kwargs['moe_top_k']
        # config.moe_group = kwargs['moe_group']
        # config.moe_group = None
        config.moe_world_size = kwargs['moe_world_size']

        # modelForLoad = GPT2LMHeadModel.from_pretrained("gpt2",config=config_load)
        modelForLoad = GPT2LMHeadModel(config=config_load, moe_group = kwargs['moe_group'])
        if config.moe_num_experts == 0:
            return modelForLoad,tokenizer
        # tokenizer = AutoTokenizer.from_pretrained("cwh/gpt2-medium-finetuned-wikitext2")
        # model = AutoModelForCausalLM.from_pretrained("cwh/gpt2-medium-finetuned-wikitext2")
        # mymoe = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
        mymoe = GPT2LMHeadModel(config=config, moe_group = kwargs['moe_group'])
        # print(modelForLoad.state_dict().keys(),mymoe.state_dict().keys())
        mymoeParam = mymoe.state_dict()
        bertParam = modelForLoad.state_dict()
        # original weight = ['transformer.h.0.mlp.c_fc.weight', 'transformer.h.0.mlp.c_fc.bias',
        #  'transformer.h.0.mlp.c_proj.weight', 'transformer.h.0.mlp.c_proj.bias']
        # desity weight = ['transformer.h.11.moe_linear.experts.15.htoh4.weight', 'transformer.h.11.moe_linear.experts.15.htoh4.bias', 
        # 'transformer.h.11.moe_linear.experts.15.h4toh.weight', 'transformer.h.11.moe_linear.experts.15.h4toh.bias',]
        # original_layer_normal = ['transformer.h.0.ln_2.weight', 'transformer.h.0.ln_2.bias']
        # desitny weight = ['transformer.h.11.moe_linear.layer_norm.weight', 'transformer.h.11.moe_linear.layer_norm.bias',]
        bertLayerLength=12
        # copy linear weight, bias and layernormal
        for layer in range(config.n_layer):
            for expert_id in range(config.moe_num_experts):
                mymoeParam['transformer.h.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.weight'] = bertParam['transformer.h.'+str(layer)+'.mlp.c_fc.weight'].T.unsqueeze(0).detach().clone()
                mymoeParam['transformer.h.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.bias'] = bertParam['transformer.h.'+str(layer)+'.mlp.c_fc.bias'].unsqueeze(0).detach().clone()
                mymoeParam['transformer.h.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.weight'] = bertParam['transformer.h.'+str(layer)+'.mlp.c_proj.weight'].T.unsqueeze(0).detach().clone()
                mymoeParam['transformer.h.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.bias'] = bertParam['transformer.h.'+str(layer)+'.mlp.c_proj.bias'].unsqueeze(0).detach().clone()
            mymoeParam['transformer.h.'+str(layer)+'.moe_linear.layer_norm.weight'] = bertParam['transformer.h.'+str(layer)+'.ln_2.weight'].detach().clone()
            mymoeParam['transformer.h.'+str(layer)+'.moe_linear.layer_norm.bias'] = bertParam['transformer.h.'+str(layer)+'.ln_2.bias'].detach().clone() 
        mymoe.load_state_dict(mymoeParam)
        return mymoe, tokenizer
    else:
        raise Exception('Error: no such a model named {}'.format(kwargs['model_name']))

