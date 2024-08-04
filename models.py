import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from peft import LoraConfig, TaskType
from peft import get_peft_model
import numpy as np

class ALBERT(nn.Module):
    def __init__(self,tokenizer_name='albert-base-v2',model_name='albert-base-v2',device='cpu'):
        super(ALBERT, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def forward(self,sentences):
        encoded_input = self.tokenize(sentences).to(self.device)
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def tokenize(self,sentences):
        tokens=self.tokenizer(list(sentences), padding=True, truncation=True, return_tensors='pt')
        return tokens

class TimeEncode(torch.nn.Module):
    # INDUCTIVE REPRESENTATION LEARNING ON TEMPORAL GRAPHS
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()

        time_dim = expand_dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        # print(self.basis_freq.shape) #[50]
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
        self.gru = nn.GRU(time_dim, time_dim, batch_first=True)
        self.comb = nn.Linear(in_features=time_dim*2,out_features=time_dim,bias=True)
        # print(self.phase.shape)#[50]

    def forward(self,rs, ts):
        length = [len(i) for i in ts]
        pad_ts = nn.utils.rnn.pad_sequence(ts,batch_first=True,padding_value=-1)
        map_ts = pad_ts.unsqueeze(-1) * self.basis_freq  # [N, L, time_dim]
        map_ts += self.phase
        harmonic = torch.cos(map_ts)
        packed_ts = nn.utils.rnn.pack_padded_sequence(harmonic,length,batch_first=True,enforce_sorted=False)
        _,ts = self.gru(packed_ts)
        ts = self.comb(torch.concatenate((rs,ts.squeeze(0)),dim=-1))
        return ts

class TKG_Model(nn.Module):
    def __init__(self,tokenizer_name,model_name,num_rels,device,h_dim=768):
        super(TKG_Model,self).__init__()
        # self.encoder = SentenceTransformer(tokenizer_name,model_name,device)
        self.encoder = ALBERT(tokenizer_name,model_name,device)

        self.h_dim = h_dim
        self.device = device
        self.time_encoder = TimeEncode(self.h_dim).to(device)
        weight = torch.ones(num_rels,requires_grad=True) * 0.5
        self.weight = nn.Parameter(weight).to(self.device)

    def forward(self,sentences,timestamps):
        sentence_embdings = self.encoder(sentences)

        timestamps = [torch.tensor(t).cuda(device=self.device) for t in timestamps]
        sentence_embdings = self.time_encoder(sentence_embdings.squeeze(),timestamps)

        return sentence_embdings