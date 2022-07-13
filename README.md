# Learning to Generate Textual Adversarial Examples.

This repository contains the codes and the resources for our paper: Learning to Generate Textual Adversarial Examples.

## Requirements

* datasets==1.5.0
* pytorch==1.8.1
* scikit-learn==0.20.3
* nltk==3.5
* transformers==4.5.0
* sentence-transformers==1.1.0
* language-tool-python==2.6.1

## Resources

### Victim Models

You can download the victim models from this [link](https://drive.google.com/file/d/1z1Z-5o7xWea7PkZFdY5FK-gtNF-JwtQq/view?usp=sharing), and place the extracted files in `resources/victim_models`. The directory structure should be like:

```
├── resources
│   ├── victim_models
│   │   ├── bert-imdb
│   │   │   ├── config.json
│   │   │   ├── pytorch_model.bin
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.txt
│   │   ├── bert-qqp
│   │   │   └── ...
│   │   ├── bert-snli
│   │   │   └── ...
│   │   ├── bert-yelp
│   │   │   └── ...
│   │   ├── lstm-imdb
│   │   │   └── ...
│   │   └── lstm-yelp
│   │       └── ...
│   └── ...
└── ...
```

The single-text input datasets have both BERT and LSTM as victim models, and the text-pair input datasets have only BERT as victim models.

### Pretrained Language Models

We use the pretrained BERT model in the two proposed attack networks, and use GPT-2 to evaluate the perplexity. The pretrained models can be downloaded from HuggingFace:

* BERT: https://huggingface.co/bert-base-uncased
* GPT-2: https://huggingface.co/gpt2

The model files should be placed in `resources/encoder_models`. The directory structure should be like:

```
├── resources
│   ├── encoder_models
│   │   ├── bert
│   │   │   ├── config.json
│   │   │   ├── pytorch_model.bin
│   │   │   ├── tokenizer.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── vocab.txt
│   │   │   └── ...
│   │   └── gpt2
│   │       ├── config.json
│   │       ├── merges.txt
│   │       ├── pytorch_model.bin
│   │       ├── tokenizer.json
│   │       ├── vocab.json
│   │       └── ...
│   └── ...
└── ...
```

### Datasets

The original datasets are in `resources/datasets/{victim_model}_original`, where the `{victim_model}` should be replaced by `bert` or `lstm`. 

To construct the training data for the two networks, run the following command:

```
python construct_training_data.py [--device DEVICE]
                                  [--dataset_path DATASET_PATH]
                                  [--output_path OUTPUT_PATH]
                                  [--model_path MODEL_PATH]
                                  [--gpt_path GPT_PATH]
```

Options:

```
--device: the device used in PyTorch, e.g., cpu or cuda
--dataset_path: the original dataset path, e.g., resources/datasets/bert_original/imdb.json
--output_path: the generated training dataset path, e.g., resources/datasets/bert_train/imdb.json
--model_path: the victim model path, e.g., resources/victim_models/bert-imdb
--gpt_path: the GPT-2 model path, e.g., resources/encoder_models/gpt2
```

For example, to generate training dataset for BERT victim model on the IMDB dataset:

```
python construct_training_data.py --device cuda --dataset_path resources/datasets/bert_original/imdb.json --output_path resources/datasets/bert_train/imdb.json --model_path resources/victim_models/bert-imdb --gpt_path resources/encoder_models/gpt2
```

This step may take a long time, and we provide generated training data in this [link](https://drive.google.com/file/d/1icr-5XeeXbFHR_YJjyipSTMTHKcWMhbS/view?usp=sharing).

Finally, the directory structure should be like:

```
├── resources
│   ├── datasets
│   │   ├── bert_original
│   │   │   ├── imdb.json
│   │   │   ├── qqp.json
│   │   │   ├── snli.json
│   │   │   └── yelp.json
│   │   ├── bert_train
│   │   │   ├── imdb.json
│   │   │   ├── qqp.json
│   │   │   ├── snli.json
│   │   │   └── yelp.json
│   │   ├── lstm_original
│   │   │   ├── imdb.json
│   │   │   └── yelp.json
│   │   └── lstm_train
│   │       ├── imdb.json
│   │       └── yelp.json
│   └── ...
└── ...
```

## Running

### Training

#### Word Ranking Network

To train the word ranking network, run the following command:

```
python attack.py train_word_ranking_module [--device DEVICE]
                                           [--encoder_path ENCODER_PATH]
                                           [--train_data_path TRAIN_DATA_PATH]
                                           [--module_path MODULE_PATH]
                                           [--sim_threshold SIM_THRESHOLD]
                                           [--pos_top_k POS_TOP_K]
                                           [--neg_top_k NEG_TOP_K]
                                           [--ppl_proportion PPL_PROPORTION]
                                           [--hidden_dim HIDDEN_DIM]
                                           [--num_classes NUM_CLASSES]
                                           [--dropout DROPOUT]
                                           [--num_epochs NUM_EPOCHS]
                                           [--batch_size BATCH_SIZE]
```

Options:

```
--device: the device used in PyTorch, e.g., cpu or cuda
--encoder_path: the path of BERT model, e.g., resources/encoder_models/bert
--train_data_path: the training dataset path, e.g., resources/datasets/bert_train/imdb.json
--module_path: the path of the word ranking network, e.g., results/prediction/imdb/bert/word_ranking_network
--sim_threshold: the similarity threshold for training data selection, set to 0.9 in the experiment.
--pos_top_k: the proportion for selecting postive samples, set to 0.05 in the experiment.
--neg_top_k: the proportion for selecting negative samples, set to 0.50 in the experiment.
--ppl_proportion: the perplexity threshold for training data selection, set to 0.8 in the experiment.
--hidden_dim: the hidden dimension, set to 128 in the experiment.
--num_classes: the number of classes of the dataset, set to 2 for IMDB, Yelp and QQP, set to 3 for SNLI.
--dropout: the dropout rate, set to 0.5 in the experiment.
--num_epochs: the number of training epochs, set to 5 in the experiment.
--batch_size: the number of batch size, set to 5 in the experiment.
```

For example, to train the synonym selection network for BERT victim model on the Imdb dataset:

```
python attack.py train_word_ranking_module --device cuda --encoder_path resources/encoder_models/bert --train_data_path resources/datasets/bert_train/imdb.json --module_path results/prediction/imdb/bert/word_ranking_network --sim_threshold 0.9 --pos_top_k 0.05 --neg_top_k 0.50 --ppl_proportion 0.8 --hidden_dim 128 --num_classes 2 --dropout 0.5 --num_epochs 5 --batch_size 5
```

#### Synonym Selection Network

To train the synonym selection network, run the following command:

```
python attack.py train_candidate_selection_module [--device DEVICE]
                                                  [--encoder_path ENCODER_PATH]
                                                  [--train_data_path TRAIN_DATA_PATH]
                                                  [--module_path MODULE_PATH]
                                                  [--sim_threshold SIM_THRESHOLD]
                                                  [--ppl_proportion PPL_PROPORTION]
                                                  [--hidden_dim HIDDEN_DIM]
                                                  [--num_classes NUM_CLASSES]
                                                  [--num_epochs NUM_EPOCHS]
                                                  [--batch_size BATCH_SIZE]
```

Options:

```
--device: the device used in PyTorch, e.g., cpu or cuda
--encoder_path: the path of BERT model, e.g., resources/encoder_models/bert
--train_data_path: the training dataset path, e.g., resources/datasets/bert_train/imdb.json
--module_path: the path of the synonym selection network, e.g., results/prediction/imdb/bert/synonym_selection_module
--sim_threshold: the similarity threshold for training data selection, set to 0.95 in the experiment.
--ppl_proportion: the perplexity threshold for training data selection, set to 0.8 in the experiment.
--hidden_dim: the hidden dimension, set to 128 in the experiment.
--num_classes: the number of classes of the dataset, set to 2 for IMDB, Yelp and QQP, set to 3 for SNLI.
--num_epochs: the number of training epochs, set to 5 in the experiment.
--batch_size: the number of batch size, set to 5 in the experiment.
```

For example, to train the synonym selection network for BERT victim model on the Imdb dataset:

```
python attack.py train_candidate_selection_module --device cuda --encoder_path resources/encoder_models/bert --train_data_path resources/datasets/bert_train/imdb.json --module_path results/prediction/imdb/bert/synonym_selection_network --sim_threshold 0.95 --ppl_proportion 0.8 --hidden_dim 128 --num_classes 2 --num_epochs 5 --batch_size 5
```

### Evaluation

For evaluation, run the following command:

```
python attack.py test [--device DEVICE]
                      [--encoder_path ENCODER_PATH]
                      [--dataset_path DATASET_PATH]
                      [--gpt_path GPT_PATH]
                      [--hidden_dim HIDDEN_DIM]
                      [--num_classes NUM_CLASSES]
                      [--top_k TOP_K]
                      [--sim_threshold SIM_THRESHOLD]
                      [--victim_model_path VICTIM_MODEL_PATH]
                      [--candidate_selection_module_path CANDIDATE_SELECTION_MODULE_PATH]
                      [--word_ranking_module_path WORD_RANKING_MODULE_PATH]
                      [--output_path OUTPUT_PATH]
```

Options:

```
--device: the device used in PyTorch, e.g., cpu or cuda
--encoder_path: the path of BERT model, e.g., resources/encoder_models/bert
--dataset_path: the original dataset path, e.g., resources/datasets/bert_original/imdb.json
--gpt_path: the GPT-2 model path, e.g., resources/encoder_models/gpt2
--hidden_dim: the hidden dimension, set to 128 in the experiment.
--num_classes: the number of classes of the dataset, set to 2 for IMDB, Yelp and QQP, set to 3 for SNLI.
--top_k: the top_k in the synonym selection step, set to 15 in the experiment.
--sim_threshold: the similarity threshold for generating adversarial examples, set to 0.9 in the experiment.
--victim_model_path: the victim model path, e.g., resources/victim_models/bert-imdb
--candidate_selection_module_path: the path of the synonym selection network, e.g., results/prediction/imdb/bert/synonym_selection_network/pytorch_model.bin
--word_ranking_module_path: the path of the word ranking network, e.g., results/prediction/imdb/bert/word_ranking_network/pytorch_model.bin
--output_path: the output path, e.g., results/prediction/imdb/bert
```

For example, to train the synonym selection network for BERT victim model on the Imdb dataset:

```
python attack.py test --device cuda --encoder_path resources/encoder_models/bert --dataset_path resources/datasets/bert_original/imdb.json --gpt_path resources/encoder_models/gpt2 --hidden_dim 128 --num_classes 2 --top_k 15 --sim_threshold 0.9 --victim_model_path resources/victim_models/bert-imdb --candidate_selection_module_path results/prediction/imdb/bert/synonym_selection_network/pytorch_model.bin --word_ranking_module_path results/prediction/imdb/bert/word_ranking_network/pytorch_model.bin --output_path results/prediction/imdb/bert
```

### Greedy

We also provide implementation for the original greedy methods, run the following command:

```
python greedy.py [--device DEVICE]
                 [--dataset_path DATASET_PATH]
                 [--model_path MODEL_PATH]
                 [--output_path OUTPUT_PATH]
                 [--gpt_path GPT_PATH]
                 [--word_order WORD_ORDER]
                 [--sim_threshold SIM_THRESHOLD]
```

Options:

```
--device: the device used in PyTorch, e.g., cpu or cuda
--dataset_path: the original dataset path, e.g., resources/datasets/bert_original/imdb.json
--model_path: the victim model path, e.g., resources/victim_models/bert-imdb
--output_path: the output path, e.g., results/greedy/imdb/bert
--gpt_path: the GPT-2 model path, e.g., resources/encoder_models/gpt2
--word_order: the method for determining word substitution order, can be seq or text-fooler.
--sim_threshold: the similarity threshold for generating adversarial examples, set to 0.9 in the experiment.
```

For example, to generate adversarial examples in sequential word substitution order:

```
python greedy.py --device cuda --dataset_path resources/datasets/bert_original/imdb.json --model_path resources/victim_models/bert-imdb --output_path results/greedy/imdb/bert --word_order seq --sim_threshold 0.9
```

## Acknowledgement

This repository used some codes in [TextAttack](https://github.com/QData/TextAttack).