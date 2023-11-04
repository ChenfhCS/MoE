import torch
from transformers import AutoTokenizer,BertForQuestionAnswering
import numpy as np
from transformers import AutoConfig

def Create_MoE_Model_Multi_Device(config_dict):
    if config_dict['model_name'] == 'bert':
        config = AutoConfig.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        modelForLoad = BertForQuestionAnswering(config=config)
        if not config_dict['moe']:
            return modelForLoad,tokenizer
        else:
            config.moe=config_dict['moe']
            config.moe_num_expert=config_dict['num_experts']
            config.moe_world_size=config_dict['world_size']
            config.moe_group=config_dict['group']
            config.moe_top_k=config_dict['top_k']
            config.fuse_token=config_dict['fuse_token']
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
        bertLayerLength=12
        # copy linear weight, bias and layernormal
        for layer in range(bertLayerLength):
            for expert_id in range(config.num_experts):
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.intermediate.dense.weight'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.intermediate.dense.bias'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.dense.weight'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.dense.bias'].unsqueeze(0).detach().clone()
            mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.layer_norm.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.LayerNorm.weight'].detach().clone()
            mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.layer_norm.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.LayerNorm.bias'].detach().clone() 
        mymoe.load_state_dict(mymoeParam)
        return mymoe, tokenizer
    elif config_dict['model_name'] == 'xl':
        from transformers import TransfoXLForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
        config = AutoConfig.from_pretrained("transfo-xl-wt103")
        # config.num_labels = 2
        modelForLoad = TransfoXLForSequenceClassification(config=config)
        if num_experts == 0:
            return modelForLoad,tokenizer
        # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        # outputs = modelForLoad(**inputs)
        config.moe=True
        config.num_experts=num_experts
        mymoe = TransfoXLForSequenceClassification(config=config)
        print(modelForLoad.state_dict().keys(),mymoe.state_dict().keys())
        mymoeParam = mymoe.state_dict()
        bertParam = modelForLoad.state_dict()
        # original weight = ['transformer.layers.0.pos_ff.CoreNet.0.weight', 'transformer.layers.0.pos_ff.CoreNet.0.bias', 
        # 'transformer.layers.0.pos_ff.CoreNet.3.weight', 
        # 'transformer.layers.0.pos_ff.CoreNet.3.bias']
        # desity weight = ['transformer.h.11.moe_linear.experts.15.htoh4.weight', 'transformer.h.11.moe_linear.experts.15.htoh4.bias', 
        # 'transformer.h.11.moe_linear.experts.15.h4toh.weight', 'transformer.h.11.moe_linear.experts.15.h4toh.bias',]
        # original_layer_normal = ['transformer.layers.0.pos_ff.layer_norm.weight', 'transformer.layers.0.pos_ff.layer_norm.bias']
        # desitny weight = ['transformer.h.11.moe_linear.layer_norm.weight', 'transformer.h.11.moe_linear.layer_norm.bias',]
        bertLayerLength=18
        # copy linear weight, bias and layernormal
        for layer in range(bertLayerLength):
            for expert_id in range(num_experts):
                mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.weight'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.CoreNet.0.weight'].unsqueeze(0).detach().clone()
                mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.bias'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.CoreNet.0.bias'].unsqueeze(0).detach().clone()
                mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.weight'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.CoreNet.3.weight'].unsqueeze(0).detach().clone()
                mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.bias'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.CoreNet.3.bias'].unsqueeze(0).detach().clone()
            mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.layer_norm.weight'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.layer_norm.weight'].detach().clone()
            mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.layer_norm.bias'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.layer_norm.bias'].detach().clone() 
        mymoe.load_state_dict(mymoeParam)
        return mymoe, tokenizer
    elif config_dict['model_name'] == 'gpt':
        from transformers import GPT2LMHeadModel, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = AutoConfig.from_pretrained("gpt2")
        modelForLoad = GPT2LMHeadModel.from_pretrained("gpt2",config=config)
        if num_experts == 0:
            return modelForLoad,tokenizer
        # tokenizer = AutoTokenizer.from_pretrained("cwh/gpt2-medium-finetuned-wikitext2")
        # model = AutoModelForCausalLM.from_pretrained("cwh/gpt2-medium-finetuned-wikitext2")
        config.moe=True
        config.num_experts=num_experts
        mymoe = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
        print(modelForLoad.state_dict().keys(),mymoe.state_dict().keys())
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
        for layer in range(bertLayerLength):
            for expert_id in range(num_experts):
                mymoeParam['transformer.h.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.weight'] = bertParam['transformer.h.'+str(layer)+'.mlp.c_fc.weight'].T.unsqueeze(0).detach().clone()
                mymoeParam['transformer.h.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.bias'] = bertParam['transformer.h.'+str(layer)+'.mlp.c_fc.bias'].unsqueeze(0).detach().clone()
                mymoeParam['transformer.h.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.weight'] = bertParam['transformer.h.'+str(layer)+'.mlp.c_proj.weight'].T.unsqueeze(0).detach().clone()
                mymoeParam['transformer.h.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.bias'] = bertParam['transformer.h.'+str(layer)+'.mlp.c_proj.bias'].unsqueeze(0).detach().clone()
            mymoeParam['transformer.h.'+str(layer)+'.moe_linear.layer_norm.weight'] = bertParam['transformer.h.'+str(layer)+'.ln_2.weight'].detach().clone()
            mymoeParam['transformer.h.'+str(layer)+'.moe_linear.layer_norm.bias'] = bertParam['transformer.h.'+str(layer)+'.ln_2.bias'].detach().clone() 
        mymoe.load_state_dict(mymoeParam)
        return mymoe, tokenizer
    else:
        print('no model ' + config_dict['model_name'])
    print('success to load ' + config_dict['model_name'])
