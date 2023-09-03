import numpy as np
import bisect

from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable, Any

import utils


class Estimator(ABC):
  """
  The abstract class for estimators.
  """
  def __init__(self) -> None:
    self.symbols = None
    self.indices = None
    self.num_symbols = None
    self.cdf = None

  def get_upper(self, symbol: int) -> float:
    """
    Get the CDF of the current symbol's index + 1.
    If it's the last index, rounding up to 1.

    Args:
      symbol: A symbol from the stream of data.

    Return:
      Probability in the [0, 1] range.
    """

    if self.indices[symbol] == self.num_symbols - 1:
      return 1.0
    return self.cdf[self.indices[symbol] + 1]

  def get_lower(self, symbol: str) -> float:
    """
    Get the CDF of the current symbol's index.

    Args:
      symbol: A symbol from the stream of data.

    Return:
      Probability in the [0, 1) range.
    """

    return self.cdf[self.indices[symbol]]

  def get_symbol(self, probability: float) -> str:
    """
    Find the corresponding symbol from given probability range using binary
    search.

    Args:
      probability: The representative value of current probability range [a, b)
      where a <= probability < b.

    Return:
      The corresponding symbol to the probability range.
    """

    symbol_idx = bisect.bisect(self.cdf, probability) - 1
    return self.symbols[symbol_idx]

  @abstractmethod
  def get_stream(self, data: Iterable[Any]) -> Iterable[Any]:
    pass

  @abstractmethod
  def get_context(self, stream: Iterable[Any], index: int) -> None:
    pass

  @abstractmethod
  def mode(self, mode: str) -> None:
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

  def __init__(self, stream_type: str) -> None:
    super().__init__()
    self.stream_type = stream_type

  def fit(self, data: Iterable[Any]) -> None:
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

  def get_context(self, stream: Iterable[Any], index: int) -> None:
    """
    This estimator is a fixed context.
    """

    pass

  def get_stream(self, data: Iterable[Any]) -> None:
    return utils.get_stream_text(data, mode=self.stream_type)

  def mode(self, mode: str) -> None:
    pass


class AdaptiveFrequencyEstimator(Estimator):
  """
  A naive implementation of adaptive frequency counter using Fenwick tree
  or Binary Index Tree.

  At each encoding or decoding step, the estimator get the new symbol and the
  out of context symbol and updating the CDF accordingly. The complexity for
  initializing the CDF is O(DlogD), for updating and getting CDF for each
  index is O(logD) where D is number of symbols.
  TODO: Add more document
  """

  def __init__(self, stream_type: str, context_width: int):
    super().__init__()
    self.eps = 0.1
    self.stream_type = stream_type
    self.context_width = context_width

  def mode(self, mode: str) -> None:
    """
    Reset the CDF table each time called.

    Args:
      mode: Either 'encode' or 'decode'.

    Returns:
      None
    """

    if mode == 'encode':
      self.reset_cdf()

    elif mode == 'decode':
      self.reset_cdf()

    else:
      pass

  def reset_cdf(self):
    self.cdf.reset()
    
  def fit(self, data: Iterable[Any]) -> None:
    """
    Build the look-up table for the symbols in the data.

    Args:
      data: The data that needed to be encoded.

    Returns:
      None
    """
    stream, _ = self.get_stream(data)
    self.symbols = sorted(set(stream))
    self.num_symbols = len(self.symbols)
    self.cdf = self.FenwickTreeCDF(self.num_symbols, self.eps)
    self.indices = {s: idx for idx, s in enumerate(self.symbols)}

  def get_context(self, stream: Iterable[Any], index: int) -> None:
    """
    Get the context and updating the probabilities accordingly.
    When invoked, the function gets the new symbol in the window context and
    the out of context symbol and updates their count.

    Args:
      stream: The data stream from the get_stream() function.
      index: The index of the being processed symbol.

    Returns:
      None.
    """

    if not stream or index < 0:
      return
    new_symbol = stream[index]
    ooc_symbol = stream[index-self.context_width] if \
      (index >= self.context_width) else None

    self.update_cdf(new_symbol, 1)
    if ooc_symbol:
      self.update_cdf(ooc_symbol, -1)

  def update_cdf(self, symbol: str, change: int) -> None:
    """
    Get the index of symbol and update it in the counting tree.
    The CDF table has the length of self.num_symbols + 1. CDF[i] is the
    cumulative probability from the 0th symbol to the (i-1)th symbol.

    Args:
      symbol: The symbol that have changed frequency.
      change: The changed amount.

    Returns:
      None
    """
    symbol_idx = self.indices[symbol]
    self.cdf.update_count(symbol_idx + 1, change)

  def get_stream(self, data):
    return utils.get_stream_text(data, mode=self.stream_type)

  class FenwickTreeCDF:
    """
    An inner class employing Fenwick tree (or BIT) for implementing the
    frequency table. This algorithm allows the update and query operator to
    perform both in the logarithmic time.

    This class can be invoked as an iterable. For example, FenwickTreeCDF()[
    10] will return the cumulative distribution of the first 10 symbols (
    0th -> 9th).
    """

    def __init__(self, num_symbols: int, eps: float = 0.1) -> None:
      """

      Args:
        num_symbols: Number of unique symbols in the data stream.
        eps: An arbitrarily small number to make sure no 2+ symbols have the
          same cumulative probability.

      Returns:
        None.
      """

      self.num_symbols = num_symbols
      self.eps = eps
      self.bit = None
      self.reset()

    def __getitem__(self, x: int) -> float:
      """
      Make the object list-like.

      Args:
        x: A 0-based index.

      Returns:
        Cumulative probability of the x-th index.
      """

      count = self.query_count(x)
      total = self.query_count(self.num_symbols)
      return count / total

    def __len__(self) -> int:
      return self.num_symbols + 1

    def reset(self) -> None:
      """
      Initializing the Fenwick tree in linear time.

      Returns:
        None
      """
      self.bit = np.zeros(self.num_symbols + 1)
      for i in range(self.num_symbols):
        self.bit[i+1] += self.eps
        if i + (i & -i) <= self.num_symbols:
          self.bit[i + (i & -i)] += self.bit[i]

    def update_count(self, x: int, value: int) -> None:
      """
      Update the frequency of the counter for the (x-1)-th symbol with a
      'value' amount along with the cumulative counter for all symbols after
      that in O(logD).

      Args:
        x: The 0-based index of the symbol that changes in frequency.
        value: The changed amount.

      Returns:
        None
      """

      while x <= self.num_symbols:
        self.bit[x] += value
        x += x & -x

    def query_count(self, x: int) -> int:
      """
      Query the cumulative frequency of the (x-1)-th symbol.

      Args:
        x: The 0-based index of the symbol.

      Returns:
        Cumulative frequency from the 0th to the (x-1)th symbol.
      """

      count = 0
      while x > 0:
        count += self.bit[x]
        x -= x & -x
      return count