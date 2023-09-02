import numpy as np
import bisect

from abc import ABC, abstractmethod
from collections import Counter

import arithmetic_coding.utils as utils


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
  A naive implementation of adaptive frequency counter. At each encoding or
  decoding step, the estimator get the context as a window and re-calculate
  the CDF of all the symbols. For this reason, the complexity is O(
  N*window_length), where N is number of symbols.
  """

  def __init__(self, stream_type, context_width):
    super().__init__()
    self.eps = 0.1
    self.stream_type = stream_type
    self.context_width = context_width

  def mode(self, mode):
    pass

  def fit(self, data):
    stream, length = self.get_stream(data)
    self.symbols = sorted(set(stream))
    self.num_symbols = len(self.symbols)
    self.indices = {s: idx for idx, s in enumerate(self.symbols)}
    self.update_cdf([])

  def get_context(self, stream, index):
    if index < self.context_width:
      context = stream[:index]
    else:
      context = stream[index - self.context_width:index]
    self.update_cdf(context)
    return context

  def update_cdf(self, context):
    counter = Counter(context)
    csum = [0]
    for symbol in self.symbols:
      csum.append(counter.get(symbol, self.eps) + csum[-1])
    self.cdf = [x / csum[-1] for x in csum]

  def get_stream(self, input):
    return utils.get_stream_text(input, mode=self.stream_type)
