import torch
from torch import nn

from models.lang_class import LanguageClassifier

#GRU's are better with less data, as we have, and a simpler version of the lstm
class GRULanguageClassifier(LanguageClassifier):

    def __init__(self, num_chars, embedding_dim=12, num_out_clases=3, num_layers=3):
        super().__init__(num_chars, num_layers=num_layers, embedding_dim=embedding_dim, num_out_clases=num_out_clases)
        #https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        #embeddings are trainable vector per character
        self.embeddings = nn.Embedding(num_chars, self.embedding_dim)

        self.gru = nn.GRU(self.embedding_dim, self.embedding_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True) #bidirectional makes it read left-to-right and right-to-left
        #gru is like a simpler lstm, a type of recurrent neural network
        #outputs 48 features
        # *4 because cat first+last and bidirectional
        self.output1 = nn.Linear(self.embedding_dim * 4, self.embedding_dim * 2) #linear layer that maps first and last element
        self.act = nn.Sigmoid() #standard [0-1] activation function
        self.output2 = nn.Linear(self.embedding_dim * 2, self.embedding_dim)  # *2 because bidirectional
        self.output3 = nn.Linear(self.embedding_dim, num_out_clases) #the output layer

    def forward(self, sent: str):
        chars = list(sent)
        # print("chars:", chars)
        char_ids: torch.Tensor = self.get_char_ids(chars).view(1, -1) #get the character ids in a tensor
        embs = self.embeddings(char_ids) #embed the character tensor to get a sequence of feature vectors
        #print("embs:", embs.size())

        out, _ = self.gru(embs, self.initHidden())
        #print("after gru:", out.size())
        #we have to add the first and last feature vectors
        first, last = out[:,0,:].view(-1, 1, 2 * self.embedding_dim), out[:,-1,:].view(-1, 1, 2* self.embedding_dim) #.view is the pytorch equivalent of reshape
        #after reading the feature vectors left to right (and creating a new sequence of feature vectors) and right to left (again producing another
        # sequence of feature vectors), it concatenaates them along the feature dimensions, and then concatenates the first and last feature vectors, which
        #are then passed into the linear network, to be dimensionally reduced to the number of output classes (3)
        #print("first:", first.size())
        out = torch.cat([first, last], dim=2)
        #print("out aft cat:", out.size())

        out = self.act(self.output1(out))
        #print("out l1:", out.size())
        out = self.act(self.output2(out))
        #print("out l2:", out.size())
        out = self.output3(out).view(-1, self.num_out_clases)
        #print("out l3:", out.size())
        #raise Exception()
        return out

    def initHidden(self):
        # *2 because bidirectional, reads left to right
        hidden = torch.zeros(self.num_layers * 2, 1, self.embedding_dim).to(self.device).detach()
        return hidden