
from typing import List

import torch
from torch import nn




class LanguageClassifier(nn.Module):

    def __init__(self, num_chars, embedding_dim=40, num_out_clases=3, num_layers=6):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_out_clases = num_out_clases #english, afrikaans, or dutch
        self.num_chars = num_chars #number of unique characters
        self.char_id_map = {}
        self.next_char_id = 0

        self.soft = nn.Softmax()


    @property
    def device(self):
        return next(self.parameters()).device

    def get_char_ids(self, chars: List[str]) -> torch.Tensor:
        """adds cls token to the beginning of the sequence"""
        # cls_id = self.long.config.se
        ids = []
        for c in chars:
            if c in self.char_id_map:
                ids.append(self.char_id_map[c])
            else:
                ids.append(self.next_char_id)
                self.char_id_map[c] = self.next_char_id
                self.next_char_id += 1
        return torch.tensor(ids).to(self.device)