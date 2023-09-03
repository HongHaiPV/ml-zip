import numpy as np
import bisect

from abc import ABC, abstractmethod
from collections import Counter

import utils


class Estimator(ABC):

  def __init__(self):
    self.symbols = None
    self.indices = None
    self.num_symbols = 0
    self.cdf = None

  def get_upper(self, symbol):
    """
    Get the CDF of the current symbol's index + 1.
    If it's the last index, rounding up to 1.

    Args:
      symbol: A symbol from the stream of data.
      context: Not used.

    Return:
      Probability in the [0, 1] range.
    """

    if self.indices[symbol] == self.num_symbols - 1:
      return 1.0
    return self.cdf[self.indices[symbol] + 1]

  def get_lower(self, symbol):
    """
    Get the CDF of the current symbol's index.

    Args:
      symbol: A symbol from the stream of data.
      context: Not used.

    Return:
      Probability in the [0, 1) range.
    """

    return self.cdf[self.indices[symbol]]

  def get_symbol(self, probability):
    """
    Find the corresponding symbol from given probability range using binary
    search.

    Args:
      probability: The representative value of current probability range [a, b)
      where a <= probability < b.
      context: Not used.

    Return:
      The corresponding symbol to the probability range.
    """
    symbol_idx = bisect.bisect(self.cdf, probability) - 1
    return self.symbols[symbol_idx]

  @abstractmethod
  def get_stream(self, input):
    pass

  @abstractmethod
  def get_context(self, stream, index):
    pass

  @abstractmethod
  def mode(self, mode):
    """
    Set the current mode for the estimator. It's either for encoding or
    decoding.

    Args:
      mode: Encode or Decode

    Returns:
      None.
    """
    pass


class FrequencyEstimator(Estimator):
  """
  A simple static estimator using frequency table.
  """

  def __init__(self, stream_type):
    super().__init__()
    self.stream_type = stream_type

  def fit(self, data):
    """
    Learn the data's frequency.
    """
    stream, length = self.get_stream(data)
    counter = Counter(stream)
    self.symbols = sorted(counter.keys())
    self.num_symbols = len(self.symbols)
    self.indices = {s: idx for idx, s in enumerate(self.symbols)}
    probability = np.array([counter[s] for s in self.symbols])
    probability = probability / sum(counter.values())
    self.cdf = np.array(
      [probability[:idx].sum() for idx in range(self.num_symbols)])

  def get_context(self, stream, index):
    """
    This estimator is a fixed context.
    """
    pass

  def get_stream(self, input):
    return utils.get_stream_text(input, mode=self.stream_type)

  def mode(self, mode):
    pass


class AdaptiveFrequencyEstimator(Estimator):
  """
  A naive implementation of adaptive frequency counter using Fenwick tree
  or Binary Index Tree. At each encoding or decoding step, the estimator get
  the new symbol and the out of context symbol and updating the CDF accordingly.
  The complexity for initializing the CDF is O(DlogD), for updating and
  getting CDF for each index is O(logD) where D is number of symbols.
  TODO: Add more document
  """

  def __init__(self, stream_type, context_width):
    super().__init__()
    self.eps = 0.1
    self.stream_type = stream_type
    self.context_width = context_width

  def mode(self, mode):
    if mode == 'encode':
      self.reset_cdf()

    elif mode == 'decode':
      self.reset_cdf()

    else:
      pass

  def reset_cdf(self):
    self.cdf.reset_cdf()
    
  def fit(self, data):
    stream, _ = self.get_stream(data)
    self.symbols = sorted(set(stream))
    self.num_symbols = len(self.symbols)
    self.cdf = self.FenwickTree(self.num_symbols, self.eps)
    self.indices = {s: idx for idx, s in enumerate(self.symbols)}

  def get_context(self, stream, index):
    if not stream or index < 0:
      return
    new_symbol = stream[index]
    ooc_symbol = stream[index-self.context_width]\
                 if index >= self.context_width else None

    self.update_cdf(new_symbol, 1)
    if ooc_symbol:
      self.update_cdf(ooc_symbol, -1)

  def update_cdf(self, symbol, change):
    symbol_idx = self.indices[symbol]
    old_val = self.cdf.query_count(symbol_idx)
    self.cdf.update_count(symbol_idx, change)

  def get_stream(self, input):
    return utils.get_stream_text(input, mode=self.stream_type)

  class FenwickTree:
    
    def __init__(self, num_symbols, eps):
      self.num_symbols = num_symbols
      self.eps = eps
      self.bit = None
      self.reset_cdf()

    def reset_cdf(self):
      # TODO: Linear initializing
      self.bit = np.zeros(self.num_symbols + 1)
      for i in range(self.num_symbols):
        self.update_count(i, self.eps)

    def update_count(self, x, value):
      x += 1
      while x <= self.num_symbols:
        self.bit[x] += value
        x += x & -x

    def query_count(self, x):
      x += 1
      count = 0
      while x > 0:
        count += self.bit[x]
        x -= x & -x
      return count

    def __getitem__(self, x):
      count = self.query_count(x-1)
      total = self.query_count(self.num_symbols-1)
      return count / total
    
    def __len__(self):
      return self.num_symbols + 1