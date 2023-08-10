from abc import ABC, abstractmethod

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

class FrequencyEstimator(Estimator):
  """
  A simple static estimator using frequency table.
  """

  def __init__(self, stream):
    self.counter = Counter(stream)
    total = sum(self.counter.values())

    self.symbols = sorted(self.counter.keys())

    self.index = {s:idx for idx, s in enumerate(self.symbols)}

    self.probability = np.array([self.counter[s] for s in self.symbols])
    self.probability = self.probability/total
    self.cdf = np.array([self.probability[:idx].sum() for idx in range(len(self.symbols))])
    self.cdf = np.append(self.cdf, [1.0])

  def get_upper(self, symbol, context):
    if self.index[symbol] == len(self.symbols)-1:
      return 1
    return self.cdf[self.index[symbol] + 1]

  def get_lower(self, symbol, context):
    return self.cdf[self.index[symbol]]

  def get_prob(self, symbol, context):
    return self.probability[self.index[symbol]]

  def get_symbol(self, prob, context):
    """
    Find the symbol from given probability range
    """
    assert prob <= 1, 'probability must <= 1'
    first = 0
    last = len(self.symbols)

    while last - first > 1:
      middle = (last+first) // 2
      # print(prob, middle, self.cdf[middle])
      if prob < self.cdf[middle]:
        last = middle
      else:
        first = middle
    assert first+1 == last
    
    # Possible problem with float precision
    if not self.cdf[first] <= prob <= self.cdf[first+1]:
      raise ValueError('Probability not in range.')
      
    # print('Result: {}'.format(self.symbols[first]))
    return self.symbols[first]