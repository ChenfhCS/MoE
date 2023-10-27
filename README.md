## Introduction

An distributed MoE system based on FastMoE. 

## Contents
* [Contents](#contents)
* [Installation](#installation)
* [Usage](#usage)
  * [Transformer-XL](#transformer-xl)
  * [GPT-2](#gpt-2)

## Installation

### Prerequisites
* Pytorch >= 1.10.0
* CUDA >= 10
* [FastMoE](https://github.com/laekov/fastmoe) == 1.1.0

If the distributed expert feature is enabled, NCCL with P2P communication
support, typically versions `>=2.7.5`, is needed. 

### Installing
```
git clone https://github.com/ChenfhCS/MoE.git
cd MoE/my_examples/
```
#### 1. Replace the modification into fastmoe
Change `path/to/MoE/my_example/fmoe` and `path/to/fmoe` to your path in ``scripts/fmoe_update.sh``, and then
```
bash scripts/fmoe_update.sh
```

## Usage

### Transformer-XL
#### 1. Download dataset
```
cd transformer-xl
bash scripts/getdata.sh
```

#### 2. Run example
```
bash scripts/run_enwik8_base_moe.sh train --work_dir=works/
```

### GPT-2
### 1. Download dataset
Before training GPT-2 model, corpus dataset should be prepared. We recommend to build your own corpus by using [Expanda](https://github.com/affjljoo3581/Expanda). Instead, training module requires tokenized training and evaluation datasets with their vocabulary file.

```
cd data/ && mkdir gpt2_wiki && cd gpt2_wiki
```
Download expanda:
```
pip install expanda
```
Then, download Wikipedia dump file from here. In this example, we are going to test with part of the [wiki](https://dumps.wikimedia.org/enwiki/). Download the file through the browser, move to workspace/src and rename to wiki.xml.bz2. Instead, run below code:
```
wget -O src/wiki.xml.bz2 https://dumps.wikimedia.org/enwiki/20230801/enwiki-20230801-pages-articles11.xml-p6899367p7054859.bz2
```
After downloading the dump file, we need to setup the configuration file. Create `expanda.cfg` file and write the below:
```
[expanda.ext.wikipedia]
num-cores           = 6

[tokenization]
unk-token           = <unk>
control-tokens      = <s>
                      </s>
                      <pad>

[build]
input-files         =
    --expanda.ext.wikipedia     src/wiki.xml.bz2
```
Now we are ready to build! Run Expanda by using:
```
expanda build
```

You may encounter a problem with:
```
TypeError: Can’t convert <tokenizers.trainers.BpeTrainer object at 0x7f8641325570> to Sequence
```

Just fix it by changing the following code in your `tokenizer.py` file:
```
tokenizer.train(trainer=trainer, files=["big.txt"])
```

### 2. Install gpt2
Install gpt2 by importing this module to the python path:
```
export PYTHONPATH=/path/to/MoE/my_examples/GPT2/src
```
or you can add the above command to `~/.bashrc` file and do

```
source ~/.bashrc
```

### 3. Run example
```
bash scripts/run_got_moe.sh train --work_dir=works/
```
