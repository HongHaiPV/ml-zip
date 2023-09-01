import numpy as np
import bisect

from abc import ABC, abstractmethod
from collections import Counter

import arithmetic_coding.utils as utils

class Estimator(ABC):

  @abstractmethod
  def get_upper(self, symbol, context):
    pass

  @abstractmethod
  def get_lower(self, symbol, context):
    pass

  @abstractmethod
  def get_symbol(self, probability_range, context):
    pass

  @abstractmethod
  def get_context(self, stream):
    pass
    
  @abstractmethod
  def get_stream(self, input, stream_type='char'):
    pass

class FrequencyEstimator(Estimator):
  """
  A simple static estimator using frequency table.
  """

  def __init__(self, stream_type):
    self.stream_type = stream_type
    
    
  def fit(self, data):
    """
    Learn the data's frequency.
    """
    stream, length = self.get_stream(data)
    self.counter = Counter(stream)
    self.total = sum(self.counter.values())
    self.symbols = sorted(self.counter.keys())
    self.index = {s:idx for idx, s in enumerate(self.symbols)}
    self.probability = np.array([self.counter[s] for s in self.symbols])
    self.probability = self.probability/self.total
    self.cdf = np.array([self.probability[:idx].sum() for idx in range(len(self.symbols))])
    self.cdf = np.append(self.cdf, [1.0])


  def get_context(self, stream):
    """
    This estimator is a fixed context.
    """

    pass

  def get_stream(self, input):

    return utils.get_stream_text(input, mode=self.stream_type)

  def get_upper(self, symbol, context):
    """
    Get the CDF of the current symbol's index + 1.
    If it's the last index, rounding up to 1.

    Args:
      symbol: A symbol from the stream of data.
      context: Not used.

    Return:
      Probability in the [0, 1] range.
    """

    if self.index[symbol] == len(self.symbols)-1:
      return 1
    return self.cdf[self.index[symbol] + 1]

  def get_lower(self, symbol, context):
    """
    Get the CDF of the current symbol's index.

    Args:
      symbol: A symbol from the stream of data.
      context: Not used.

    Return:
      Probability in the [0, 1) range.
    """

    return self.cdf[self.index[symbol]]

  def get_symbol(self, probability, context):
    """
    Find the corresponding symbol from given probability range using binary
    search.

    Args:
      probability: The representative value of current probability range [a, b]
      where a <= probability < b.
      context: Not used.

    Return:
      The corresponding symbol to the probability range.
    """

    symbol_idx = bisect.bisect(self.cdf, probability)-1
    return self.symbols[symbol_idx]


class AdaptiveEstimator(Estimator):


  def get_upper(self, symbol, context):
    pass

  @abstractmethod
  def get_lower(self, symbol, context):
    pass

  @abstractmethod
  def get_symbol(self, probability_range, context):
    pass

  @abstractmethod
  def get_context(self, stream):
    pass
    
  @abstractmethod
  def get_stream(self, input, stream_type='char'):
    pass