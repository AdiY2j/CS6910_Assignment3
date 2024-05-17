# CS6910_Assignment3

Goal of this assignment is threefold: (i) learn how to model sequence-to-sequence learning problems using Recurrent Neural Networks (ii) compare different cells such as vanilla RNN, LSTM and GRU (iii) understand how attention networks overcome the limitations of vanilla seq2seq model

In this assignment, we experimented with a sample of the Aksharantar dataset released by AI4Bharat. This dataset contains pairs of the following form: (x, y) = (ajanabee,अजनबी)

## Hyperparameter Searching

- No. of epochs
- Hidden Size
- Embedding Size
- Optimizer
- Learning Rate
- Bidirectional
- Attention
- No. of encoder & decoder layers
- Cell Type
- Dropout
- Batch Size
- Teacher forcing ratio

## Best Model Configuration 

- No. of epochs : 20
- Hidden Size : 256
- Embedding Size : 128
- Optimizer : nadam
- Learning Rate : 1e-3
- Bidirectional : True
- Attention : True
- No. of encoder & decoder layers : 3
- Cell Type : LSTM 
- Dropout : 0.2
- Batch Size : 256
- Teacher Forcing Ratio : 0.6

## Python Script for Training Model

I designed a python script (train.py) to train above Seq2Seq with different parameters. Inorder to execute it just run below command :

```
python train.py --wandb_project "myprojectname" --wandb_entity "myname" --epochs 20 --batch_size 256 --num_layer 3 --hidden_size 256 --embedding_size 128 --learning_rate 0.001 --optimizer "nadam" --bidirectional True --attention True --drop_out 0.2 --cell_type "LSTM" --teacher_forcing 0.6
```

OR

To just run best model just execute :

```
python train.py
```

## Command Line Arguments

| Argument         | Default Value | Required | Type | Choices                     | Description                                                     |
|------------------|---------------|----------|------|-----------------------------|-----------------------------------------------------------------|
| `--wandb_project` | DL_Assignment_3 | False | str |                             | Project name used to track experiments in Weights & Biases dashboard |
| `--wandb_entity`  | cs23m009 | False | str |                             | Wandb Entity used to track experiments in the Weights & Biases dashboard |
| `--num_layers`    | 3             | False | int | 1, 2, 3                     | Number of layers in encoder and decoder                         |
| `--epochs`        | 10            | False | int |                             | Number of epochs to train model                                 |
| `--batch_size`    | 128           | False | int |                             | Batch size used to train model                                  |
| `--hidden_size`   | 256           | False | int |                             | Hidden size used to train model                                 |
| `--embedding_size`| 128           | False | int |                             | Embedding size used to train model                              |
| `--cell_type`     | LSTM          | False | str | RNN, GRU, LSTM              | Cell Type choices                                               |
| `--bidirectional` | True          | False | bool| True, False                 | Bidirectional Value                                             |
| `--attention`     | False         | False | bool| True, False                 | Perform Attention                                               |
| `--learning_rate` | 0.001         | False | float|                             | Learning rate used to optimize model parameters                 |
| `--drop_out`      | 0.2           | False | float|                             | Dropout Value                                                   |
| `--teacher_forcing` | 0.5         | False | float|                             | Teacher Force Value                                             |
| `--optimizer`     | nadam         | False | str | sgd, rmsprop, adam, nadam   | Activation Function choices                                     |


