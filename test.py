from datasets import load_dataset
import time
from torch.optim.lr_scheduler import LambdaLR

import torch
import numpy as np
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,SwitchTransformersForConditionalGeneration,BertForQuestionAnswering
import pickle
import wandb

import numpy as np
# import evaluate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler,DataCollatorForSeq2Seq
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
from transformers import DefaultDataCollator
# import datasets
import nltk
import evaluate
import collections
from transformers import AutoConfig

def save_model(model,name):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(dir_path + '/pth/', "%s_checkpoint.bin" % name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    print("Saved model checkpoint to [DIR: /home/ubuntu/SwitchTransformer/pth/]")

def Create_MoE_Model(model_name, num_experts):
    if model_name == 'bert':
        config = AutoConfig.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        modelForLoad = BertForQuestionAnswering(config=config)
        if num_experts == 0:
            return modelForLoad,tokenizer
        config.moe=True
        config.num_experts=num_experts
        mymoe = BertForQuestionAnswering(config=config)
        print(modelForLoad.state_dict().keys(),mymoe.state_dict().keys())
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
            for expert_id in range(num_experts):
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.intermediate.dense.weight'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.intermediate.dense.bias'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.dense.weight'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.dense.bias'].unsqueeze(0).detach().clone()
            mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.layer_norm.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.LayerNorm.weight'].detach().clone()
            mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.layer_norm.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.LayerNorm.bias'].detach().clone() 
        mymoe.load_state_dict(mymoeParam)
        return mymoe, tokenizer
    elif model_name == 'xl':
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
    elif model_name == 'gpt':
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
        print('no model ' + model_name)
    print('success to load ' + model_name)

def train_GPT_MoE():
    from transformers import DataCollatorWithPadding
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    
    dataset = load_dataset("samsum")
    metric = evaluate.load("rouge")

    model,tokenizer = Create_MoE_Model('gpt',2) # AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")


    def preprocess_function(examples):
        # inputs = [doc for doc in examples['dialogue']]
        model_inputs = tokenizer(examples['dialogue'], padding="max_length", max_length=1024, truncation=True)

        # Setup the tokenizer for targets
        labels = tokenizer(text_target=examples["summary"], padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    
    batch_size=1
    # dataset = load_dataset("yelp_review_full")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))

    tokenized_datasets = dataset.map(preprocess_function,  batched=True)
    tokenized_datasets.set_format("torch")
    # tokenized_datasets = tokenized_datasets.remove_columns(["valid"])
    tokenized_datasets = tokenized_datasets.remove_columns(["dialogue"])
    tokenized_datasets = tokenized_datasets.remove_columns(["id"])
    tokenized_datasets = tokenized_datasets.remove_columns(["summary"])
    
    train_dataset = tokenized_datasets["train"].shuffle(seed=42) # .select(range(1000))
    eval_dataset = tokenized_datasets["test"]# .shuffle(seed=42) # .select(range(1000))



    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=None,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=5e-05,
                                betas=(0.9,0.999),
                                eps=1e-08)
    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # model.train()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    progress_bar = tqdm(range(num_training_steps))
    model = model.to(device)
    best_acc = 0
    model_name='gpt'

    wandb.init(    # set the wandb project where this run will be logged
    project="switch-8-samsum",
    name='gpt',
    # track hyperparameters and run metadata
    
    config={
    "learning_rate": 5e-05,
    "architecture": "gpt",
    "dataset": "samsum",
    "epochs": num_epochs,
    }
    )

    for epoch in range(num_epochs):
        model.train()
        step = 0
        loss_all = 0
        for batch in train_dataloader:
            # break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            step += 1
            wandb.log({'batch_loss': loss_all/step})
            break
        # dict_router = {}
        # index = 0
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model.generate(batch['input_ids'])# (**batch)
                # outputs = model(**batch)
            # logits = outputs.logits
            # predictions = torch.argmax(logits, dim=-1)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            # dict_router[index]=(outputs.encoder_router_logits,outputs.decoder_router_logits)
            # index += 1
            # with open("./experiment/router_dict_finetune_o.pkl", "wb") as file:
            #     # print(score_dict)
            #     pickle.dump(dict_router, file)
            break
        result = metric.compute()

        wandb.log({'loss': loss_all/step, 'rouge1': result['rouge1']})
        if best_acc < result['rouge1']:
            save_model(model,model_name)
            best_acc = result['rouge1']
        # break
    # print(result)
    wandb.finish()
    del model
    del dataset
    del tokenizer
def train_Bert_MoE():
    def compute_metrics(start_logits, end_logits, features, examples):

        n_best = 20
        max_answer_length = 30

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)

    model,tokenizer = Create_MoE_Model('bert',1)

    max_length = 384
    stride = 128


    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs
    
    datasets = load_dataset("squad")
    # raw_datasets  = raw_datasets.train_test_split(test_size=0.2)
    # raw_datasets  = raw_datasets.rename_column("test", "validation")
    metric = evaluate.load("squad")
    # tokenized_squad = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    train_dataset = datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )
    eval_dataset = datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=datasets["validation"].column_names,
    )
    validation_dataset = eval_dataset.remove_columns(["example_id", "offset_mapping"])

    data_collator = DefaultDataCollator()

    batch_size=4
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(validation_dataset, collate_fn=data_collator, batch_size=batch_size)
    num_epochs = 1
    model_name="bert" # config1[some_args]['model']
    # metric = evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad")
    wandb.init(    # set the wandb project where this run will be logged
    project="switch-8-samsum",
    name='bert', # config1[some_args]['model'],
    # track hyperparameters and run metadata
    
    config={
    "learning_rate": 3e-5,
    "architecture": model_name,
    "dataset": "samsum",
    "epochs": num_epochs,
    }
    )

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=3e-5)
    # ,
                                # betas=(0.9,0.999),
                                # eps=1e-08)
    # num_epochs = 8
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    progress_bar = tqdm(range(num_training_steps))
    model = model.to(device)
    best_acc = 0
    
    
    for epoch in range(num_epochs):
        model.train()
        step = 0
        loss_all = 0
        for batch in train_dataloader:
            # break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            step += 1
            wandb.log({'batch_loss': loss_all/step})
            # break
        # dict_router = {}
        # index = 0
        model.eval()
        # question_answerer = pipeline("question-answering", model=model)
        start_logits = []
        end_logits = []
        # accelerator.print("Evaluation!")
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)]
        end_logits = end_logits[: len(validation_dataset)]
        # metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
        metrics = compute_metrics(start_logits, end_logits, eval_dataset, datasets["validation"])
        # {'exact_match': 83.0, 'f1': 88.25}
        wandb.log({'loss': loss_all/step, 'exact_match':metrics['exact_match'],'f1':metrics['f1']}) # 'rouge1': result['rouge1']})
        if best_acc < metrics['f1']:
            save_model(model,model_name)
            best_acc = metrics['exact_match']
    

    wandb.finish()
    del model
    del datasets
    del tokenizer
def train_xl_MoE():
    from transformers import DataCollatorWithPadding
    dataset = load_dataset("glue", "cola")
    # dataset = load_dataset("imdb",split="train[10:20]")
    # dataset  = dataset.train_test_split(test_size=0.2)
    
    model,tokenizer = Create_MoE_Model('xl',2)
    tokenizer.model_max_length = 250
    print(model)
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], padding="max_length",truncation=True)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))
    tokenized_datasets = dataset.map(preprocess_function,  batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric = evaluate.load("accuracy")
    model_name = 'xl'
    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     predictions = np.argmax(predictions, axis=1)
    #     return metric.compute(predictions=predictions, references=labels)
    # id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    # label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    tokenized_datasets.set_format("torch")
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
    tokenized_datasets = tokenized_datasets.remove_columns(["idx"])
    train_batch_size = 1
    eval_batch_size = 1
    train_dataloader = DataLoader(tokenized_datasets["train"].shuffle(seed=42), collate_fn=data_collator,shuffle=True, batch_size=train_batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=eval_batch_size)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=5e-5,
                                betas=(0.9,0.999),
                                eps=1e-08)
    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    progress_bar = tqdm(range(num_training_steps))
    model = model.to(device)
    best_acc = 0
    

    wandb.init(    # set the wandb project where this run will be logged
    project="switch-8-samsum",
    name='xl',
    # track hyperparameters and run metadata
    
    config={
    "learning_rate": 5e-05,
    "architecture": "xl",

    "dataset": "samsum",
    "epochs": 8,
    }
    )

    for epoch in range(num_epochs):
        model.train()
        step = 0
        loss_all = 0
        for batch in train_dataloader:
            # break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            step += 1
            wandb.log({'batch_loss': loss_all/step})
            break

        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            break
        metrics = metric.compute()
        wandb.log({'loss': loss_all/step, 'acc':metrics['accuracy']}) # 'rouge1': result['rouge1']})
        if best_acc < metrics['accuracy']:
            save_model(model,model_name)
            best_acc = metrics['accuracy']
    

    wandb.finish()
    del model
    del dataset
    del tokenizer
train_GPT_MoE()
train_Bert_MoE()
train_xl_MoE()