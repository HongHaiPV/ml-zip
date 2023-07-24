import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl

torch.manual_seed(1)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class LanguageModel:

  def __init__(self,
         modeler,
         context_width,
         level,
         model_config):
    self.level = level
    self.context_width = context_width
    self.modeler = modeler
    self.embed_size = model_config['embed_size']
    self.hidden_size =  model_config['hidden_size']
    self.batch_size = model_config['batch_size']
    self.lr = model_config['lr']
    self.current_context = None
    self.cummulative_distribution = None

  def fit(self, data, epochs):
    
    if self.level == 'char':
      self.vocab = sorted(set(data))
    else:
      data = data.split(' ')
      self.vocab = sorted(set(data))

    self.vocab.append('<PAD>')
    self.index = {symbol:i for i, symbol in enumerate(self.vocab)}

    self.model = self.modeler(len(self.vocab), self.context_width-1,
            self.embed_size, self.hidden_size).to(device)
    
    ngrams = [['<PAD>']*i + list(data[:self.context_width-i]) for i in 
          range(self.context_width-1, 0, -1)]
    ngrams += [list(data[i:i+self.context_width]) for i in range(len(data) 
          - self.context_width + 1)]
    losses = []
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    for epoch in range(epochs):
      total_loss = 0
      for i in range(0, len(ngrams), self.batch_size):
        batch_ngram = ngrams[i:i+self.batch_size]
        
        contexts = [ngram[:-1] for ngram in batch_ngram]
        targets = [ngram[-1] for ngram in batch_ngram]
        
        context_idxs = torch.tensor([[self.index[s] for s in context] 
                  for context in contexts], dtype=torch.long).to(device)
        target_idxs = torch.tensor([self.index[target] for target in 
                  targets], dtype=torch.long).to(device)
        
        self.model.zero_grad()
        log_probs = self.model(context_idxs)
        loss = loss_fn(log_probs, target_idxs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if i % (100*self.batch_size) == 0:
          print('Epoch: {} Sample: {} Loss: {}'.format(epoch, i, loss.item()))
      losses.append(total_loss)
           
  def save(self, path):
    file = open(path + '.pkl', 'wb')
    pkl.dump(self, file)
    file.close()
  
  def get_context(self, stream, idx):
    
    if len(stream) < self.context_width - 1:      
      return ['<PAD>']*(self.context_width - len(stream) - 1) + stream
    
    return stream[-(self.context_width - 1 - idx):idx]

  def calculate_cdistribution(self, context):
    
    if self.current_context == context:
      return self.cummulative_distribution
    else:
      context = torch.tensor([self.index[s] for s in context],
            dtype=torch.long).to(device)
      distribution = torch.exp(self.model(context)).cpu().data.numpy()
      if len(distribution) == 0:
        raise
        distribution = [1/len(self.vocab) for _ in range(len(self.vocab))]
      else:
        distribution = distribution[0]
      
      distribution /= distribution.sum()
      
      self.cummulative_distribution = [0 for i in range(len(self.vocab))]
      for i in range(len(self.vocab)):
        if i == 0:
          self.cummulative_distribution[i] = 0
        else:
          self.cummulative_distribution[i] = distribution[i] + 
                        self.cummulative_distribution[i-1]
      
      self.cummulative_distribution.append(1.0)
      
    
  def get_upper(self, symbol, context):
    self.calculate_cdistribution(context)
    symbol_idx = self.index[symbol]
    return self.cummulative_distribution[symbol_idx+1]

  def get_lower(self, symbol, context):
    self.calculate_cdistribution(context)
    symbol_idx = self.index[symbol]
    return self.cummulative_distribution[symbol_idx]

  def get_prob(self, symbol, context):
    return self.get_upper(symbol, context) - self.get_lower(symbol, context)

  def get_symbol(self, prob, context):
    self.calculate_cdistribution(context)
    first = 0
    last = len(self.vocab)

    while last - first > 1:
      middle = (last+first) // 2
      if prob < self.cummulative_distribution[middle]:
        last = middle
      else:
        first = middle
    assert first+1 == last
    
    return self.vocab[first]

class NGramModel(nn.Module):
  def __init__(self, vocab_size, context_size, embed_size, hidden_size):
    super().__init__()
    self.double()
    self.embed_size = embed_size
    self.context_size = context_size
    
    self.embed = nn.Embedding(vocab_size, embed_size)
    self.linear1 = nn.Linear(embed_size * self.context_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, vocab_size)

  def forward(self, inputs):
    embeds = self.embed(inputs)
    out1 = torch.sigmoid(self.linear1(embeds))
    out2 = self.linear2(out1)
    log_probs = F.log_softmax(out2, dim=1)
    return log_probs
  
  
class LSTMSimple(nn.Module):
  def __init__(self, vocab_size, context_size, embed_size, hidden_size):
    super().__init__()
    self.double()
    self.context_size = context_size
    self.embed_size = embed_size
    self.lstm_size = hidden_size
    self.num_layers = 2
    self.vocab_size = vocab_size
    
    self.embed = nn.Embedding(self.vocab_size, self.embed_size)
    self.lstm = nn.LSTM(self.embed_size,
              self.lstm_size,
              self.num_layers,
              dropout=0)
    self.linear = nn.Linear(self.lstm_size, self.vocab_size)
    

  def forward(self, inputs, prev_state):
    embeds = self.embed(inputs)
    # print('embed output shape', embeds.shape)
    output, state = self.lstm(embeds, prev_state)
#     print(state)
    log_prob = F.log_softmax(self.linear(output), dim=1)
    return log_prob, state
  
  def init_state(self, sequence_length):
    return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
        torch.zeros(self.num_layers, sequence_length, self.lstm_size))
  
class LSTMLanguageModel(LanguageModel):
  def __init__(self,
         modeler,
         context_width,
         level,
         model_config):
    
    super().__init__(modeler,
             context_width,
             level,
             model_config)
    
  def fit(self, data, epochs):
    
    if self.level == 'char':
      self.vocab = sorted(set(data))
    else:
      data = data.split(' ')
      self.vocab = sorted(set(data))

    self.vocab.append('<PAD>')
    self.index = {symbol:i for i, symbol in enumerate(self.vocab)}

    self.model = self.modeler(len(self.vocab), self.context_width-1, 
            self.embed_size, self.hidden_size).to(device)
    
    ngrams = [['<PAD>']*i + list(data[:self.context_width-i]) for i in 
            range(self.context_width-1, 0, -1)]
    ngrams += [list(data[i:i+self.context_width]) for i in range(len(data) 
              - self.context_width + 1)]
    
    
    losses = []
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    for epoch in range(epochs):
      state_h, state_c = self.model.init_state(self.model.context_size)
      total_loss = 0
      for i in range(0, len(ngrams), self.batch_size):
        batch_ngram = ngrams[i:i+self.batch_size]
        
        contexts = [ngram[:-1] for ngram in batch_ngram]
        targets = [ngram[1:] for ngram in batch_ngram]
        
        context_idxs = torch.tensor([[self.index[s] for s in context] 
                          for context in contexts], dtype=torch.long).to(device)
        target_idxs = torch.tensor([[self.index[t] for t in target] 
                          for target in targets], dtype=torch.long).to(device)
        
        self.model.zero_grad()
        log_probs, (state_h, state_c) = self.model(context_idxs, (state_h, state_c))
        
#         print(log_probs.shape, state_h.shape, state_c.shape)
        state_h = state_h.detach()
        state_c = state_c.detach() 
        
#         print(log_probs.shape)
        loss = loss_fn(log_probs.transpose(1,2), target_idxs)
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item()
        
        if i % (100*self.batch_size) == 0:
          print('Epoch: {} Sample: {} Loss: {}'.format(epoch, i, loss.item()))
      losses.append(total_loss)
  
  def calculate_cdistribution(self, context):
    context_size = len(context)
    if self.current_context == context:
      return self.cummulative_distribution
    else:
      context = torch.tensor([[self.index[s] for s in context]]
                  , dtype=torch.long).to(device)
      state_h, state_c = self.model.init_state(context_size)
      # print(state_h.shape, state_c.shape)
      # print('context shape', context.shape)
      log_prob, (state_h, state_c) = self.model(context, (state_h, state_c))
      distribution = torch.exp(log_prob).cpu().data.numpy()
      # print(distribution.shape)
      
      if len(distribution) == 0:
        raise
        distribution = [1/len(self.vocab) for _ in range(len(self.vocab))]
      else:
        distribution = distribution[0][-1]
      
      distribution /= distribution.sum()
      
      self.cummulative_distribution = [0 for i in range(len(self.vocab))]
      for i in range(len(self.vocab)):
        if i == 0:
          self.cummulative_distribution[i] = 0
        else:
          self.cummulative_distribution[i] = distribution[i]
                                             + self.cummulative_distribution[i-1]
      
      self.cummulative_distribution.append(1.0)