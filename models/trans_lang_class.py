import torch
from torch import nn
from transformers import LongformerModel, LongformerConfig

from models.lang_class import LanguageClassifier


class TransformerLanguageClassifier(LanguageClassifier):

    def __init__(self, num_chars, embedding_dim=40, num_layers=6, num_out_clases=3):
        super().__init__(num_chars, embedding_dim=embedding_dim)

        self.long = LongformerModel(LongformerConfig(attention_window=10, vocab_size=num_chars + 1,
                                                     hidden_size=embedding_dim, num_hidden_layers=num_layers,
                                                     num_attention_heads=1, intermediate_size=embedding_dim*2,
                                                     type_vocab_size=1))

        self.output = nn.Linear(embedding_dim, num_out_clases)

    def forward(self, sent: str):
        chars = list(sent)
        #print("chars:", chars)
        char_ids: torch.Tensor = self.get_char_ids(chars).view(1, -1)
        #print("ids:", char_ids)
        hidden = self.long(input_ids=char_ids, return_dict=True)["last_hidden_state"]
        #print("hidden:", hidden.size())
        out = hidden[:,0,:]
        #print("out:", out.size())
        out = self.soft(self.output(out))
        #print("out:", out)
        return out
