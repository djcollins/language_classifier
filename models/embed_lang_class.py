from abc import ABC

from torch import nn

from models.lang_class import LanguageClassifier


class EmbeddingLanguageClassifier(LanguageClassifier, ABC):

    def __init__(self, num_chars, embedding_dim=40, num_out_clases=3, num_layers=6):
        super().__init__(num_chars, num_layers=num_layers, embedding_dim=embedding_dim, num_out_clases=num_out_clases)
        self.embeddings = nn.Embedding(num_chars, self.embedding_dim)