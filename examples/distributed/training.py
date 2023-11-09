import time
import pickle
import numpy as np
import nltk
import evaluate
import collections
import wandb
import os
import time
dir_path = os.path.dirname(os.path.realpath(__file__))

import torch
# Load model directly
from transformers import (AutoModelForSeq2SeqLM, DefaultDataCollator, get_scheduler, DataCollatorForSeq2Seq,
                            )
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
# import evaluate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modeling import Create_MoE_Model, save_model
from fmoe.distributed import DistributedGroupedDataParallel as DDP

# train transformer-xl
def train_xl_MoE(**kwargs):
    device = kwargs['device']
    model = kwargs['model']
    tokenizer = kwargs['tokenizer']
    train_batch_size = kwargs['train_batch_size']
    eval_batch_size = kwargs['eval_batch_size']
    log_interval = kwargs['log_interval']
    eval_interval = kwargs['eval_interval']
    num_epochs = kwargs['num_epochs']
    logger = kwargs['logger']
    use_wandb = kwargs['use_wandb']
    local_rank = kwargs['local_rank']
    moe_sync_group = kwargs['moe_sync_group']
    dist = kwargs['dist']

    from transformers import DataCollatorWithPadding
    dataset = load_dataset("glue", "cola")
    # dataset = load_dataset("imdb",split="train[10:20]")
    # dataset  = dataset.train_test_split(test_size=0.2)

    tokenizer.model_max_length = 250
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
    tokenized_datasets.set_format("torch")
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
    tokenized_datasets = tokenized_datasets.remove_columns(["idx"])
    train_dataloader = DataLoader(tokenized_datasets["train"].shuffle(seed=42), collate_fn=data_collator,shuffle=True, batch_size=train_batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=eval_batch_size)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=5e-5,
                                betas=(0.9,0.999),
                                eps=1e-08)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    if local_rank == 0:
        progress_bar = tqdm(range(num_training_steps))
    best_acc = 0

    if use_wandb is True:
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

    # ddp
    if dist:
        model = DDP(model, device_ids=[local_rank], moe_sync_group = moe_sync_group)
        model._sync_params()

    for epoch in range(num_epochs):
        model.train()
        step = 0
        loss_all = 0
        elapsed_all = 0
        loss_log = 0
        elapsed_log = 0
        for batch in train_dataloader:
            # break
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_start = time.time()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            elapsed_all += time.time() - batch_start
            step += 1
            if use_wandb is True:
                wandb.log({'batch_loss': loss_all/step})
            # break

            if local_rank == 0:
                progress_bar.set_description('Epoch {} | Loss {:.2f} | acc {:.2f} | mean batch time {:.2f}'.format(
                                            epoch, (loss_all/step), best_acc, (elapsed_all/step)*1000))
                progress_bar.update(1)

            if step % eval_interval == 0:
                model.eval()
                for idx, batch in enumerate(eval_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(predictions=predictions, references=batch["labels"])
                    if idx >= 10:
                        break
                metrics = metric.compute()
                if use_wandb is True:
                    wandb.log({'loss': loss_all/step, 'acc':metrics['accuracy']}) # 'rouge1': result['rouge1']})
                if best_acc < metrics['accuracy']:
                    save_model(model,model_name)
                    best_acc = metrics['accuracy']

    if use_wandb is True:
        wandb.finish()
    del model
    del dataset
    del tokenizer

# train bert
def train_Bert_MoE(**kwargs):
    device = kwargs['device']
    model = kwargs['model']
    tokenizer = kwargs['tokenizer']
    train_batch_size = kwargs['train_batch_size']
    eval_batch_size = kwargs['eval_batch_size']
    log_interval = kwargs['log_interval']
    eval_interval = kwargs['eval_interval']
    num_epochs = kwargs['num_epochs']
    logger = kwargs['logger']
    use_wandb = kwargs['use_wandb']
    local_rank = kwargs['local_rank']
    moe_sync_group = kwargs['moe_sync_group']
    dist = kwargs['dist']
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

    batch_size=train_batch_size
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(validation_dataset, collate_fn=data_collator, batch_size=batch_size)
    num_epochs = num_epochs
    model_name="bert" # config1[some_args]['model']
    # metric = evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad")
    if use_wandb:
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

    progress_bar = tqdm(range(num_training_steps))
    model = model.to(device)
    best_acc = 0
    
    # ddp
    if dist:
        model = DDP(model, device_ids=[local_rank], moe_sync_group = moe_sync_group)
        model._sync_params()

    for epoch in range(num_epochs):
        model.train()
        step = 0
        loss_all = 0
        loss_log = 0
        elapsed_all = 0
        elapsed_log = 0

        for batch in train_dataloader:
            # break
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_start = time.time()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            elapsed_all += time.time() - batch_start
            step += 1
            if use_wandb:
                wandb.log({'batch_loss': loss_all/step})
            if local_rank == 0:
                progress_bar.set_description('Epoch {} | Loss {:.2f} | acc {:.2f} | mean batch time {:.2f}'.format(
                                            epoch, (loss_all/step), best_acc, (elapsed_all/step)*1000))
                progress_bar.update(1)
        # dict_router = {}
        # index = 0
            if step % eval_interval == 0:
                model.eval()
                # question_answerer = pipeline("question-answering", model=model)
                start_logits = []
                end_logits = []
                # accelerator.print("Evaluation!")
                stop_batch = 10
                for idx, batch in enumerate(eval_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)

                    start_logits.append(outputs.start_logits.cpu().numpy())
                    end_logits.append(outputs.end_logits.cpu().numpy())
                    if idx >= stop_batch:
                        break
                start_logits = np.concatenate(start_logits)
                end_logits = np.concatenate(end_logits)
                # start_logits = start_logits[: len(validation_dataset)]
                # end_logits = end_logits[: len(validation_dataset)]
                start_logits = start_logits[: stop_batch+1]
                end_logits = end_logits[: stop_batch+1]
                # metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
                metrics = compute_metrics(start_logits, end_logits, eval_dataset, datasets["validation"])
                # {'exact_match': 83.0, 'f1': 88.25}
                if use_wandb:
                    wandb.log({'loss': loss_all/step, 'exact_match':metrics['exact_match'],'f1':metrics['f1']}) # 'rouge1': result['rouge1']})
                if best_acc < metrics['f1']:
                    save_model(model,model_name)
                    best_acc = metrics['exact_match']
    
    if use_wandb:
        wandb.finish()
    del model
    del datasets
    del tokenizer

# train gpt
def train_GPT_MoE(**kwargs):
    device = kwargs['device']
    model = kwargs['model']
    tokenizer = kwargs['tokenizer']
    train_batch_size = kwargs['train_batch_size']
    eval_batch_size = kwargs['eval_batch_size']
    log_interval = kwargs['log_interval']
    eval_interval = kwargs['eval_interval']
    num_epochs = kwargs['num_epochs']
    logger = kwargs['logger']
    use_wandb = kwargs['use_wandb']
    local_rank = kwargs['local_rank']
    moe_sync_group = kwargs['moe_sync_group']
    dist = kwargs['dist']
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

    def preprocess_function(examples):
        # inputs = [doc for doc in examples['dialogue']]
        model_inputs = tokenizer(examples['dialogue'], padding="max_length", max_length=1024, truncation=True)

        # Setup the tokenizer for targets
        labels = tokenizer(text_target=examples["summary"], padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    
    batch_size=train_batch_size
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
    num_epochs = num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # model.train()

    progress_bar = tqdm(range(num_training_steps))
    model = model.to(device)
    best_acc = 0
    model_name='gpt'

    if use_wandb:
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

    # ddp
    if dist:
        model = DDP(model, device_ids=[local_rank], moe_sync_group = moe_sync_group)
        model._sync_params()

    for epoch in range(num_epochs):
        model.train()
        step = 0
        loss_all = 0
        elapsed_all = 0
        loss_log = 0
        elapsed_log = 0
        for batch in train_dataloader:
            # break
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_start = time.time()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            elapsed_all += time.time() - batch_start
            step += 1
            if use_wandb:
                wandb.log({'batch_loss': loss_all/step})
            # break
            if local_rank == 0:
                progress_bar.set_description('Epoch {} | Loss {:.2f} | acc {:.2f} | mean batch time {:.2f}'.format(
                                            epoch, (loss_all/step), best_acc, (elapsed_all/step)*1000))
                progress_bar.update(1)
        # dict_router = {}
        # index = 0
            if step % eval_interval == 0:
                model.eval()
                for idx, batch in enumerate(eval_dataloader):
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
                    if idx >= 10:
                        break
                result = metric.compute()

                if use_wandb:
                    wandb.log({'loss': loss_all/step, 'rouge1': result['rouge1']})
                if best_acc < result['rouge1']:
                    save_model(model,model_name)
                    best_acc = result['rouge1']
        # break
    # print(result)
    if use_wandb:
        wandb.finish()
    del model
    del dataset
    del tokenizer
