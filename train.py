import os
import wandb
import csv
import random
import argparse
from tqdm.notebook import tqdm
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('-wp', '--wandb_project', default="DL_Assignment_3", required=False, metavar="", type=str, help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we', '--wandb_entity', default="cs23m009", required=False, metavar="", type=str, help='Wandb Entity used to track experiments in the Weights & Biases dashboard')
parser.add_argument('-n', '--num_layers', default=3, required=False, metavar="", type=int, choices= ['1','2', '3'], help='Number of layers in encoder and decoder') 
parser.add_argument('-e', '--epochs', default=10, required=False, metavar="", type=int, help='Number of epochs to train model')
parser.add_argument('-bs', '--batch_size', default=128, required=False, metavar="", type=int, help='Batch size used to train model')
parser.add_argument('-hs', '--hidden_size', default=256, required=False, metavar="", type=int, help='Hidden size used to train model')
parser.add_argument('-es', '--embedding_size', default=128, required=False, metavar="", type=int, help='Embedding size used to train model')
parser.add_argument('-c', '--cell_type', default="LSTM", required=False, metavar="", type=str, choices=['RNN', 'GRU', 'LSTM'], help="Cell Type choices: ['RNN', 'GRU', 'LSTM']")
parser.add_argument('-bd', '--bidirectional', default=True, required=False, metavar="", type=bool, choices= ["True", "False"], help='Bidirectional Value')
parser.add_argument('-at', '--attention', default=False, required=False, metavar="", type=bool, choices= ["True", "False"], help='Perform Attention')
parser.add_argument('-lr', '--learning_rate', default=0.001, required=False, metavar="", type=float, help='Learning rate used to optimize model parameters')
parser.add_argument('-dp', '--drop_out', default=0.2,  required=False, metavar="", type=float, help='Dropout Value')
parser.add_argument('-tf', '--teacher_forcing', default=0.5,  required=False, metavar="", type=float, help='Teacher Force Value')
parser.add_argument('-o', '--optimizer', default="nadam", required=False, metavar="", type=str, choices=['sgd', 'rmsprop', 'adam', 'nadam'], help="Activation Function choices: ['sgd', 'rmsprop', 'adam', 'nadam']")
args = parser.parse_args()


# Enter your wandb login key
wandb.login(key='72a114321dd97dbf11db7b15eb05b2660c2faa94')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_char, EOS_char, TEMP_char, PAD_char = 0, 1, 3, 2

# Set base path of your dataset (I am using hin subfolder)
base_path = 'aksharantar_sampled/hin/'


'''
Lang class is used to create dictionaries which maps character to index and vice-versa. 
It's referred from the blog mentioned in the report 
'''
class Lang:
    def __init__(self, name):
        self.word2count = {'<' : 1, '>' : 1, '_' : 1, '$' : 1}
        self.word2index = {'<' : 0, '>' : 1, '_' : 2, '$' : 3} 
        self.name = name
        self.index2word = {SOS_char : '<', EOS_char : '>', PAD_char : '_', TEMP_char : '$'}
        self.n_chars = 4

    def add_word(self, word):
        for c in word:
            self.add_char(c)

    def add_char(self, char):
        if char not in self.word2index: # If char not present add it in word2index and inc counter
            self.word2index[char] = self.n_chars
            self.word2count[char] = 1
            self.index2word[self.n_chars] = char
            self.n_chars += 1
        else:
            self.word2count[char] += 1 #If char already present just increment counter

'''
Function takes input as input and output language dictionary and returns pairs, max_input_length, max_output_length
'''
def prepData(dir_path, iplang, oplang):
    data = pd.read_csv(dir_path)
    data = np.array(data)

    max_ip_length = max([len(word) for word in data[:, 0]])
    
    max_op_length = max([len(word) for word in data[:, 1]])
    
    input_lang, output_lang = Lang(iplang), Lang(oplang)

    pairs = []
    for i in range(len(data)): 
        pairs.append([data[i][0],data[i][1]])

    for i in range(len(pairs)):
        input_lang.add_word(pairs[i][0])
        output_lang.add_word(pairs[i][1])

    return {'pairs' : pairs,'max_input_length' : max_ip_length,'max_target_length' : max_op_length,'input_lang' : input_lang,'output_lang' : output_lang}

''' 
This function takes input as lang object, word and max length
It appends word's index with SOS_char at the start and EOS_char at the end 
If length of this word is less than maxlength then it appends with PAD_char and returns its tensor
'''
def getWordTensor(lang, word, maxlen):
    idx = [SOS_char]
    for i in range(len(word)):
        if word[i] in lang.word2index.keys():
            idx.append(lang.word2index[word[i]])
        else:
            idx.append(TEMP_char)

    idx.append(EOS_char)
    diff = (maxlen - len(idx))
    idx.extend(([PAD_char] * diff))
    return torch.LongTensor(idx).to(device)


''' 
getTensorPairs function takes input and target word pairs as input along with their respective language type with maxlen
It returns its tensor pairs using above getWordTensor function
'''
def getTensorPairs(pairs, ip_lang, op_lang, maxlen):
    tensor_pairs = []
    for data in pairs:
        tensor_pairs.append((getWordTensor(ip_lang, data[0], maxlen), getWordTensor(op_lang, data[1], maxlen)))
    return tensor_pairs

# This function helps to convert train, valid and test data into tensors & it also calculates max length among all of them. 
def generateTensor(lang1, lang2):
    train_data = prepData(base_path + 'hin_train.csv', lang1, lang2)    
    val_data   = prepData(base_path + 'hin_valid.csv', lang1, lang2)
    test_data  = prepData(base_path + 'hin_test.csv', lang1, lang2)
    
    total_max_len = max([train_data['max_input_length'], train_data['max_target_length'], val_data['max_input_length'], val_data['max_target_length'], test_data['max_input_length'], test_data['max_target_length']]) + 2
    
    train_pairs = getTensorPairs(train_data['pairs'], train_data['input_lang'], train_data['output_lang'] , total_max_len)
    val_pairs   = getTensorPairs(val_data['pairs'], train_data['input_lang'], train_data['output_lang'], total_max_len)
    test_pairs  = getTensorPairs(test_data['pairs'], train_data['input_lang'], train_data['output_lang'], total_max_len)

    return train_pairs, val_pairs, test_pairs, train_data['input_lang'], train_data['output_lang'], total_max_len


'''
    Function to train a batch of data using the encoder-decoder model.

    Args:
    - batch_size: The size of the batch.
    - num_layers: Number of layers in the encoder and decoder.
    - inputTensor: Input tensor containing sequences of characters.
    - targetTensor: Target tensor containing sequences of characters.
    - encoder: Encoder object.
    - decoder: Decoder object.
    - enc_optimizer: Optimizer for the encoder.
    - dec_optimizer: Optimizer for the decoder.
    - criterion: Loss criterion.
    - max_len: Maximum length of sequences.
    - is_attention: Flag indicating whether attention mechanism is used.
    - tf_ratio: Teacher forcing ratio (default is 0.5).

    Returns:
    - Loss value 
'''
def train_batch(batch_size, num_layers, inputTensor, targetTensor, encoder, decoder, enc_optimizer, dec_optimizer, criterion, max_len, is_attention, tf_ratio = 0.5):
    loss = 0
    inputTensor = inputTensor.transpose(0, 1)
    targetTensor = targetTensor.transpose(0, 1)
    enc_hidden = encoder.initHidden(batch_size,num_layers)
    
    if encoder.cell_type == "LSTM":
        enc_hidden = (enc_hidden, encoder.initHidden(batch_size,num_layers))
        
    ip_len = inputTensor.size(0)
    op_len = targetTensor.size(0)
    
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()
    
    num_Dir = 1
    if encoder.bidirectional :
        num_Dir = 2
        
    enc_outputs = torch.zeros(max_len, batch_size, encoder.hidden_size * num_Dir).to(device)
        
    for i in range(ip_len):
        enc_output, enc_hidden = encoder(inputTensor[i], batch_size, enc_hidden)
        enc_outputs[i] = enc_output[0]

    dec_input = torch.LongTensor([SOS_char]*batch_size).to(device)
    dec_output = None
    dec_hidden = enc_hidden

    if random.random() < tf_ratio:
        for i in range(op_len):
            if is_attention == True:
                dec_output, dec_hidden, dec_attn = decoder(dec_input, batch_size, dec_hidden, enc_outputs.reshape(batch_size, max_len, encoder.hidden_size * num_Dir))
            else:
                dec_output, dec_hidden= decoder(dec_input, batch_size, dec_hidden)
            loss += criterion(dec_output, targetTensor[i])
            dec_input = targetTensor[i]
    else:
        for i in range(op_len):
            if is_attention == True :
                dec_output, dec_hidden, dec_attn = decoder(dec_input, batch_size, dec_hidden, enc_outputs.reshape(batch_size, max_len, encoder.hidden_size * num_Dir))
            else:
                dec_output, dec_hidden = decoder(dec_input, batch_size, dec_hidden)
            _, top_i = dec_output.data.topk(1)
            dec_input = top_i
            loss += criterion(dec_output, targetTensor[i])

    loss.backward()
    enc_optimizer.step()
    dec_optimizer.step()
    
    return loss.item() / op_len


"""
    Function to evaluate the performance of the encoder-decoder model on a given dataset.

    Parameters:
    - batch_size: The size of the batch.
    - num_layers: Number of layers in the encoder and decoder.
    - encoder: Encoder object.
    - decoder: Decoder object.
    - loader: Data loader for the dataset.
    - input_lang: Input language object.
    - output_lang: Output language object.
    - max_len: Maximum length of sequences.
    - is_attention: Flag indicating whether attention mechanism is used.
    - test: Flag indicating whether to perform testing (default is False).

    Output:
    - Accuracy of the model on the dataset.
    """
def evaluate(batch_size, num_layers, encoder, decoder, loader, input_lang, output_lang, max_len, is_attention, test=False):
    with torch.no_grad():
        num_samples = 0
        correct_ans = 0
        inputX = []
        outputY = []
        predList = []
        
        for inputWord, targetWord in loader:    
            trans_input  = inputWord.transpose(0, 1)
            trans_output = targetWord.transpose(0, 1)
            enc_hidden = encoder.initHidden(batch_size,num_layers)
            if encoder.cell_type == "LSTM":
                enc_hidden = (enc_hidden, encoder.initHidden(batch_size,num_layers))

            ip_len = trans_input.size(0)
            op_len = trans_output.size(0)

            output = Variable(torch.LongTensor(op_len, batch_size))
            
            num_Dir = 1
            if encoder.bidirectional :
                num_Dir = 2
        
            enc_outputs = torch.zeros(max_len, batch_size, encoder.hidden_size * num_Dir).to(device)

            if test:
                for i in range(inputWord.size(0)):
                    x = [input_lang.index2word[c.item()] for c in inputWord[i] if c not in [SOS_char, EOS_char, PAD_char, TEMP_char]]
                    temp_val = ''
                    for j in range(len(x)):
                        temp_val += x[j]
                    inputX.append(temp_val)
             
            for i in range(ip_len):
                enc_output, enc_hidden = encoder(trans_input[i], batch_size, enc_hidden)
                enc_outputs[i] = enc_output[0]

            dec_input = torch.LongTensor(([SOS_char] * batch_size)).to(device)
            dec_output = None
            dec_hidden = enc_hidden

            for i in range(op_len):
                if is_attention == True:
                    dec_output, dec_hidden, dec_attn = decoder(dec_input, batch_size, dec_hidden, enc_outputs.reshape(batch_size, max_len, encoder.hidden_size * num_Dir))
                else:
                    dec_output, dec_hidden = decoder(dec_input, batch_size, dec_hidden)
                _, top_i = dec_output.data.topk(1)
                output[i] = torch.cat(tuple(top_i))
                dec_input = top_i
                
            output = output.transpose(0,1)
            
            outputLen = output.size(0)

            for i in range(outputLen):
                pred = [output_lang.index2word[c.item()] for c in output[i] if c not in [SOS_char, EOS_char, PAD_char, TEMP_char]]
                y = [output_lang.index2word[c.item()] for c in targetWord[i] if c not in [SOS_char, EOS_char, PAD_char, TEMP_char]]
                num_samples += 1
                
                pred_word, output_word = '', ''
                for j in range(len(pred)):
                    pred_word += pred[j]
                for j in range(len(y)):
                    output_word += y[j]
                    
                if pred == y:
                    correct_ans += 1
                
                if test:
                    outputY.append(output_word)
                    predList.append(pred_word)
        if test:
            test_df = pd.DataFrame({"Input": inputX, "Prediction": predList, "Actual": outputY})
            test_df.to_csv("prediction_vanilla.csv")
#             data = pd.read_csv("prediction_vanilla.csv")
#             table = wandb.Table(dataframe=data)
#             wandb.log({"data": table})

    
    return correct_ans / num_samples



# This function helps to calculate validation loss 
def findValLoss(batch_size, num_layers, encoder, decoder, inputTensor, targetTensor, criterion, max_len, is_attention):
    with torch.no_grad():
        loss = 0
        inputTensor = inputTensor.transpose(0, 1)
        targetTensor = targetTensor.transpose(0, 1)
        enc_hidden = encoder.initHidden(batch_size,num_layers)
        
        if encoder.cell_type == "LSTM":
            enc_hidden = (enc_hidden, encoder.initHidden(batch_size,num_layers))

        ip_len = inputTensor.size(0)
        op_len = targetTensor.size(0)
        
        num_Dir = 1
        if encoder.bidirectional :
            num_Dir = 2
        
        enc_outputs = torch.zeros(max_len, batch_size, encoder.hidden_size * num_Dir).to(device)
        
         
        for i in range(ip_len):
            enc_output, enc_hidden = encoder(inputTensor[i], batch_size, enc_hidden)
            enc_outputs[i] = enc_output[0]

        dec_input = torch.LongTensor(([SOS_char] * batch_size)).to(device)
        dec_hidden = enc_hidden
        dec_output = None
        
        for i in range(op_len):
            if is_attention == True:
                dec_output, dec_hidden, dec_attn = decoder(dec_input, batch_size, dec_hidden, enc_outputs.reshape(batch_size, max_len, encoder.hidden_size * num_Dir))
            else:
                dec_output, dec_hidden = decoder(dec_input, batch_size, dec_hidden)
            _, top_i = dec_output.data.topk(1)
            dec_input = top_i
            loss += criterion(dec_output, targetTensor[i])

    return loss.item() / op_len


"""
    This function trains the encoder-decoder model using the specified parameters.

    Parameters:
    - batch_size: Size of each training batch.
    - num_layers: Number of layers in the encoder and decoder.
    - encoder: Encoder model.
    - decoder: Decoder model.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - learning_rate: Learning rate for optimization.
    - max_length: Maximum length of sequences.
    - epochs: Number of training epochs.
    - optimizer: Name of optimizer (choices: 'sgd', 'rmsprop', 'nadam', 'adam').
    - input_lang: Input language object.
    - output_lang: Output language object.
    - is_attention: Flag indicating whether attention mechanism is used.
    - tf_ratio: Teacher forcing ratio.

"""
def train(batch_size, num_layers, encoder, decoder, train_loader, val_loader, learning_rate, max_length, epochs, optimizer, input_lang, output_lang, is_attention, tf_ratio):
    enc_optimizer = None
    criterion = nn.CrossEntropyLoss()
    dec_optimizer = None
    
    # Initialize optimizer based on the specified optimizer name
    match optimizer:
        case "sgd":
            enc_optimizer = optim.SGD(encoder.parameters(),lr=learning_rate)
            dec_optimizer = optim.SGD(decoder.parameters(),lr=learning_rate)
        case "rmsprop":
            enc_optimizer = optim.RMSprop(encoder.parameters(),lr=learning_rate)
            dec_optimizer = optim.RMSprop(decoder.parameters(),lr=learning_rate)
        case "nadam":
            enc_optimizer = optim.NAdam(encoder.parameters(),lr=learning_rate)
            dec_optimizer = optim.NAdam(decoder.parameters(),lr=learning_rate)
        case "adam":
            enc_optimizer = optim.Adam(encoder.parameters(),lr=learning_rate)
            dec_optimizer = optim.Adam(decoder.parameters(),lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        total_train_loss = 0
        total_val_loss = 0
        train_samples = 0
        val_samples = 0

        # Training
        for ip_batch, op_batch in tqdm(train_loader):
            train_samples += 1
            loss = train_batch(batch_size, num_layers, ip_batch, op_batch, encoder, decoder, enc_optimizer, dec_optimizer, criterion, max_length, is_attention, tf_ratio)
            total_train_loss += loss
  
        total_train_loss = total_train_loss / train_samples

        # Validation
        for ip_batch, op_batch in tqdm(val_loader):
            val_samples += 1
            loss = findValLoss(batch_size, num_layers, encoder, decoder, ip_batch, op_batch, criterion, max_length, is_attention)
            total_val_loss += loss

        total_val_loss = total_val_loss / val_samples

        # Compute and log accuracy
        train_accuracy = evaluate(batch_size, num_layers, encoder, decoder, train_loader, input_lang, output_lang, max_length, is_attention)
        val_accuracy = evaluate(batch_size, num_layers, encoder, decoder, val_loader, input_lang, output_lang, max_length, is_attention)

        print('Epoch : {}, Train Loss : {}, Train Acc : {}, Val Loss : {}, Val Acc : {}'.format(epoch+1,total_train_loss, train_accuracy, total_val_loss, val_accuracy))
        
        wandb.log({'Epoch' : epoch+1, 'Train Loss' : total_train_loss, 'Train Accuracy' : train_accuracy, 'Val Loss' : total_val_loss, 'Val Accuracy' : val_accuracy})


# This function helps to test the model and return test accuracy and logs it onto wandb
def test_model(batch_size, num_layers, encoder, decoder, test_loader, input_lang, output_lang, max_length, is_attention):
    test_accuracy = evaluate(batch_size, num_layers, encoder, decoder, test_loader, input_lang, output_lang, max_length, is_attention, test=True)
    print('Test Acc : {}'.format(test_accuracy))
    wandb.log({'Test Accuracy ' : test_accuracy})

''' 
This is Encoder class of the Seq2Seq model
Params used to constructor for initialization :
- input_size: Size of the input vocabulary.
- hidden_size: Dimensionality of the hidden state.
- embedding_size: Dimensionality of the embedding space.
- num_layers: Number of layers in the encoder.
- dropout_val: Dropout probability.
- cell_type: Type of RNN cell (choices: 'RNN', 'LSTM', 'GRU').
- batch_size: Size of the batch.
- bidirectional: Flag indicating whether the encoder is bidirectional (default is False).

'''

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, num_layers, dropout_val, cell_type, batch_size, bidirectional=False):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.cell_type = cell_type
        
        self.model = None
        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.dropout = nn.Dropout(dropout_val)
        self.bidirectional = bidirectional
        
        match cell_type:
            case "RNN":
                self.model = nn.RNN(self.embedding_size, self.hidden_size, num_layers=self.num_layers, dropout=dropout_val, bidirectional=self.bidirectional)
            case "LSTM":
                self.model = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=self.num_layers, dropout=dropout_val, bidirectional=self.bidirectional)
            case "GRU":
                self.model = nn.GRU(self.embedding_size, self.hidden_size, num_layers=self.num_layers, dropout=dropout_val, bidirectional=self.bidirectional)

    def initHidden(self, batch_size, num_layers):
        num_dir = 1
        
        if self.bidirectional :
            num_dir = 2
        return torch.zeros(num_layers * num_dir, batch_size, self.hidden_size, device=device)
         
    def forward(self, input, batch_size, hidden):
        embedded = self.embedding(input).view(1,batch_size, -1)
        output, hidden = self.model(self.dropout(embedded), hidden)
        return output, hidden



''' 
This is Decoder class of the Seq2Seq model

Params used to constructor for initialization :
- hidden_size: Dimensionality of the hidden state.
- output_size: Size of the output vocabulary.
- embedding_size: Dimensionality of the embedding space.
- num_layers: Number of layers in the encoder.
- dropout_val: Dropout probability.
- cell_type: Type of RNN cell (choices: 'RNN', 'LSTM', 'GRU').
- batch_size: Size of the batch.
- bidirectional: Flag indicating whether the encoder is bidirectional (default is False).

It's forward function return : Output tensor and hidden state tensor.
'''
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_size, num_layers, dropout_val, cell_type, batch_size, bidirectional):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.cell_type = cell_type
        self.dropout = nn.Dropout(dropout_val)
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.model = None
        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.num_dir = 1

        match cell_type:
            case "RNN":
                self.model = nn.RNN(self.embedding_size, self.hidden_size, num_layers=self.num_layers, dropout=dropout_val, bidirectional=self.bidirectional)
            case "LSTM":
                self.model = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=self.num_layers, dropout=dropout_val, bidirectional=self.bidirectional)
            case "GRU":
                self.model = nn.GRU(self.embedding_size, self.hidden_size, num_layers=self.num_layers, dropout=dropout_val, bidirectional=self.bidirectional)

        if self.bidirectional :
            self.num_dir = 2

        self.out = nn.Linear(self.hidden_size * self.num_dir, self.output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        

    def forward(self, input, batch_size, hidden):
        embedded = self.embedding(input).view(1,batch_size, -1)
        embedded = F.relu(self.dropout(embedded))
        output, hidden = self.model(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        
        return output, hidden

# This is DecoderAttention class specifically designed for attention models in addition to vanilla decoder it returns attention weights
class DecoderAttention(nn.Module) :
    def __init__(self, hidden_size, output_size, embedding_size, num_layers, dropout_val, cell_type, batch_size, max_len_all, bidirectional):

        super(DecoderAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.cell_type = cell_type
        self.dropout = nn.Dropout(dropout_val)
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.max_length_word = max_len_all
        self.model = None
        self.embedding = nn.Embedding(output_size, self.embedding_size)
        self.attention_layer = nn.Linear(self.embedding_size + self.hidden_size, self.max_length_word+1)
        self.num_dir = 1

        match cell_type:
            case "RNN":
                self.model = nn.RNN(self.embedding_size, self.hidden_size, num_layers=self.num_layers, dropout=dropout_val, bidirectional=self.bidirectional)
            case "LSTM":
                self.model = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=self.num_layers, dropout=dropout_val, bidirectional=self.bidirectional)
            case "GRU":
                self.model = nn.GRU(self.embedding_size, self.hidden_size, num_layers=self.num_layers, dropout=dropout_val, bidirectional=self.bidirectional)

        if self.bidirectional :
            self.num_dir = 2
            
        self.club_attention = nn.Linear(self.embedding_size + self.hidden_size * self.num_dir, self.embedding_size)
        self.out = nn.Linear(self.hidden_size * self.num_dir, self.output_size)
        

    def forward(self, input, batch_size, hidden, encoder_outputs) :
        embedded = self.embedding(input).view(1, batch_size, -1)
        attention_weights = None
        if self.cell_type == 'LSTM' :
            attention_weights = F.softmax(self.attention_layer(torch.cat((embedded[0], hidden[0][0]), 1)), dim = 1).view(batch_size,1,self.max_length_word+1)
        else :
            attention_weights = F.softmax(self.attention_layer(torch.cat((embedded[0], hidden[0]), 1)), dim = 1).view(batch_size,1,self.max_length_word+1)

        applied_attn = torch.bmm(attention_weights, encoder_outputs).view(1,batch_size,-1)
        output = self.club_attention(torch.cat((embedded[0], applied_attn[0]), 1)).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.model(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim = 1)

        return output, hidden, attention_weights


# Source and Target Languages (Eng, Hindi)
source_lang, target_lang = 'eng', 'hin'


def main():
    wandb.init(project = args.wandb_project, entity = args.wandb_entity)
    run_name = 'cell_{}_bs_{}_lr_{}_e_{}_nl_{}_dp_{}_bi_{}_op_{}_at_{}'.format(args.cell_type, args.batch_size, args.learning_rate, args.epochs, args.num_layers, args.drop_out, args.bidirectional, args.optimizer, args.attention)
    wandb.run.name = run_name

    # Prepare Tensor pairs for train, validation, test pairs
    pairs, val_pairs, test_pairs, input_lang, output_lang, max_len  = generateTensor(source_lang, target_lang)

    # Encoder and Decoder Model     
    encoder = Encoder(input_lang.n_chars, args.hidden_size, args.embedding_size, args.num_layers, args.drop_out, args.cell_type, args.batch_size, args.bidirectional).to(device)
    decoder = Decoder(args.hidden_size, output_lang.n_chars, args.embedding_size, args.num_layers, args.drop_out, args.cell_type, args.batch_size, args.bidirectional).to(device)
    attn_decoder = DecoderAttention(args.hidden_size, output_lang.n_chars, args.embedding_size, args.num_layers, args.drop_out, args.cell_type, args.batch_size, max_len, args.bidirectional).to(device)

    # Dataloader for train, valid and test data using batch size
    train_loader = DataLoader(pairs, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_pairs, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_pairs, batch_size=args.batch_size, shuffle=False, drop_last=True)
        
    temp_decoder = decoder
    if args.attention:
        temp_decoder = attn_decoder
        max_len += 1

    # Train Model        
    train(args.batch_size, args.num_layers, encoder, temp_decoder, train_loader, val_loader, args.learning_rate, max_len, args.epochs, args.optimizer, input_lang, output_lang, args.attention, args.teacher_forcing)
    # Test Model
    test_model(args.batch_size, args.num_layers, encoder, temp_decoder, test_loader,input_lang, output_lang, max_len, args.attention)
    wandb.finish()

    

main()