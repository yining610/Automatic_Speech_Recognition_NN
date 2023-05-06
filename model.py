import torch


class LSTM_ASR(torch.nn.Module):
    def __init__(self, feature_type="discrete", input_size=64, hidden_size=256, num_layers=2,
                 output_size=28):
        super().__init__()
        assert feature_type in ['discrete', 'mfcc']
        # Build your own neural network. Play with different hyper-parameters and architectures.
        # === write your code here ===

        if feature_type == "discrete":
            vocab_size = 256 # 256 quantized 2-character labels
            self.word_embeddings = torch.nn.Embedding(vocab_size, input_size)

        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)



    def forward(self, batch_features):
        """
        :param batch_features: batched acoustic features
                             : (batch_size, seq_len)
        :return: the output of your model (e.g., log probability)
        """
        # === write your code here ===

        # embedded_batch_features: (batch_size, seq_len, input_size)
        embedded_batch_features = self.word_embeddings(batch_features)
        print(f"Shape of embedded_batch_features: {embedded_batch_features.shape}")

        # lstm_out: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(embedded_batch_features) 
        print(f"Shape of lstm_out: {lstm_out.shape}")

        # linear_out: (batch_size, seq_len, output_size)
        linear_out = self.linear(lstm_out)
        print(f"Shape of linear_out: {linear_out.shape}")
        
        # log_prob: (batch_size, seq_len, output_size)
        log_prob = torch.nn.functional.log_softmax(linear_out, dim=2)
        print(f"Shape of log_prob: {log_prob.shape}")

        return log_prob
