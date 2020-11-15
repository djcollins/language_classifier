import torch
from torch import nn

from models.embed_lang_class import EmbeddingLanguageClassifier


class LSTMLanguageClassifier(EmbeddingLanguageClassifier):

    def __init__(self, num_chars, embedding_dim=40, num_out_clases=3, num_layers=6):
        super().__init__(num_chars, num_layers=num_layers, embedding_dim=embedding_dim, num_out_clases=num_out_clases)
        self.gru = nn.GRU(self.embedding_dim, self.embedding_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)

        self.output1 = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.act = nn.Sigmoid()
        self.output2 = nn.Linear(self.embedding_dim, num_out_clases)

    def forward(self, sent: str):
        chars = list(sent)
        # print("chars:", chars)
        char_ids: torch.Tensor = self.get_char_ids(chars).view(1, -1)
        embs = self.embeddings(char_ids)
        print("embs:", embs.size())

        out, _ = self.gru(embs, self.initHidden())
        out = torch.cat()
        out = self.act(self.output1(out))
        out = self.output2(out)
        print("out:", out.size())
        return out

    def initHidden(self):
        hidden = torch.zeros(self.num_layers * 2, 1, self.embedding_dim).to(self.device)  # *2 because bidirectional
        return hidden