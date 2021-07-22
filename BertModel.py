import tensorflow_hub
from bert import tokenization

class DisasterDetector:

    def __init__(self, bert : tensorflow_hub.KerasLayer, max_sequence_length = 128, lr =\
                 0.001, epochs = 15, batch_size = 32):

        self.bert_layer = bert
        self.max_sequence_length = max_sequence_length
        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.models = []
        self.scores = {}

    def encode(self, texts):
        pass