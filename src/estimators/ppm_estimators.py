"""
Classes for prediction by partial maching estimators.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from typing import Sequence, Any
import utils
from frequency_estimators import Estimator

torch.set_printoptions(precision=10)
torch.set_default_tensor_type(torch.DoubleTensor)


class PPMEstimator(Estimator):
  """
  Generic class for PPM estimators, take in the model class and model configs.
  """

  def __init__(self, stream_type, context_width, model_type, model_configs):
    super().__init__()
    self.stream_type = stream_type
    self.context_width = context_width
    self.model_configs = model_configs
    self.model_type = model_type
    self.model = None
    self.padding = model_configs.get('padding', '<PAD>')

  def get_context(self, stream: Sequence[Any], index: int) -> None:
    context = utils.rolling_window_context(stream, self.context_width,
                                           self.padding, index + 1)
    context_ids = [self.indices[i] for i in context]
    self.cdf = self.model.get_cdf(context_ids)

  def load_train_data(self, stream, stream_length):
    """
    TODO: Populate this one.

    Args:
      stream:
      stream_length:

    Returns:

    """
    windows = [utils.rolling_window_context(stream, self.context_width,
                                            self.padding, i) for i in
               range(stream_length + 1)]

    contexts = [[self.indices[s] for s in context] for context in
                windows[:-1]]
    targets = [[self.indices[s] for s in label] for label in windows[1:]]
    return contexts, targets

  def get_stream(self, data):
    """
    Using the text stream function, split data based on characters or words.

    Args:
      data: The original data that need to be encoded.

    Returns:
      A sequence of symbols.
      The length of that sequence.
    """

    return utils.get_stream_text(data, mode=self.stream_type)

  def fit(self, data):
    stream, length = self.get_stream(data)
    self.symbols = sorted(set(stream) | {self.padding})
    self.num_symbols = len(self.symbols)
    self.indices = {s: idx for idx, s in enumerate(self.symbols)}
    self.model_configs['vocab_size'] = self.num_symbols
    self.model = self.model_type(self.model_configs)
    self.model.vocab_size = self.num_symbols
    contexts, targets = self.load_train_data(stream, length)
    self.model.fit(contexts, targets)

  def mode(self, mode: str) -> None:
    """
    Reset the CDF table each time called.

    Args:
      mode: Either 'encode' or 'decode'.

    Returns:
      None
    """
    self.model.state_h, self.model.state_c = self.model.init_state()


class LSTM(nn.Module):
  """
  LSTM language model.
  """
  def __init__(self, configs):
    super().__init__()

    self.embed_size = configs.get('embed_size', 128)
    self.hidden_size = configs.get('hidden_size', 128)
    self.num_layers = configs.get('num_layers', 2)
    self.vocab_size = configs.get('vocab_size', 256)
    self.context_width = configs.get('context_width', 128)
    self.batch_size = configs.get('batch_size', 128)
    self.epochs = configs.get('epochs', 120)
    self.lr = configs.get('lr', 1e-4)
    self.state_h, self.state_c = self.init_state()

    self.device = torch.device('cuda') \
                  if torch.cuda.is_available() else torch.device('cpu')

    self.embed = nn.Embedding(self.vocab_size, self.embed_size)
    self.lstm = nn.LSTM(input_size=self.embed_size,
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        dropout=0)
    self.linear = nn.Linear(self.hidden_size, self.vocab_size)

  def forward(self, inputs, prev_state):
    embeds = self.embed(inputs)
    output, state = self.lstm(embeds, prev_state)
    log_prob = F.log_softmax(self.linear(output), dim=-1)
    return log_prob, state

  def init_state(self):
    """
    Initialize the hidden state at the beginning each epoch.
    TODO: Populate this one.
    Returns:

    """
    return (torch.zeros(self.num_layers, self.context_width, self.hidden_size),
            torch.zeros(self.num_layers, self.context_width, self.hidden_size))

  def fit(self, contexts, targets):
    """
    TODO: Populate this one.
    Args:
      contexts:
      targets:

    Returns:

    """
    n_samples = len(targets)
    losses = []
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(self.parameters(), lr=self.lr)

    self.train()

    for epoch in range(self.epochs):
      total_loss = 0
      state_h, state_c = self.init_state()
      for batch_ids in range(0, n_samples, self.batch_size):
        optimizer.zero_grad()

        batch_contexts = contexts[batch_ids: batch_ids + self.batch_size]
        batch_targets = targets[batch_ids: batch_ids + self.batch_size]

        contexts_tensor = torch.tensor(batch_contexts, dtype=torch.long).to(
          self.device)
        targets_tensor = torch.tensor(batch_targets, dtype=torch.long).to(
          self.device)
        log_probs, (state_h, state_c) = self.forward(contexts_tensor,
                                                     (state_h, state_c))
        state_h = state_h.detach()
        state_c = state_c.detach()

        loss = loss_fn(log_probs.transpose(1, 2), targets_tensor)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        if batch_ids % (100 * self.batch_size) == 0:
          print('Epoch: {} Sample: {} Loss: {}'.format(epoch, batch_ids,
                                                       loss.item()))
      losses.append(total_loss)

  def get_cdf(self, context):
    """
    TODO: Populate this one.

    Args:
      context:

    Returns:

    """
    self.eval()
    context = torch.tensor([context]).to(self.device)
    self.state_h, self.state_c = self.state_h, self.state_c
    log_prob, (self.state_h, self.state_c) = self(context,
                                                  (self.state_h, self.state_c))
    distribution = torch.exp(log_prob[0, -1]).cpu().data.numpy()
    cdf = np.cumsum(distribution)
    cdf = np.concatenate([[0], cdf])
    return cdf
