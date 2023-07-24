from collections import Counter
import numpy as np

PRECISION = 32
MSB_MASK = 1 << PRECISION-1
SMSB_MASK = 1 << PRECISION-2
SIGN_MASK = (1 << PRECISION) - 1

class ArithmeticCoding:

  def __init__(self, model, stream_type='char'):
    self._lower = 0x0
    self._upper = SIGN_MASK
    self._underflow_bits = 0
    self._model = model
    self._stream_type = stream_type

  def get_context(self, stream):
    """
    Get window context from stream and adding padding symbols (<PAD>) if needed.

    Args:
      stream: stream of symbols, i.e. bytes, characters or words.

    Returns:
      List of symbols with special <PAD> characters with length required from
      the language model.
    """
    
    if not hasattr(self._model, 'context_width'):
      return None
  
    if len(stream) < self._model.context_width - 1:      
      return ['<PAD>']*(self._model.context_width - len(stream) - 1) + stream

    return stream[-(self._model.context_width - 1):]
    

  def streaming(self, inp):
    """
    Turn inputs into stream of symbols

    Args:
      inp: a string

    Returns:
      List of symbols, i.e. bytes, characters, or words.
    """

    # Character model
    if self._stream_type == 'char':
      return list(inp), len(inp)
    
    # Word model
    elif self._stream_type == 'word':
      return inp.split(' '), len(inp.split(' '))

  def append_bits(self, encoded):
    """
    Turn inputs into stream of symbols

    Args:
      inp: a string

    Returns:
      List of symbols, i.e. bytes, characters, or words.
    """
    while True:
      if (self._upper & MSB_MASK) == (self._lower & MSB_MASK):
        # Add most significant bit to output
        encoded.append((self._upper & MSB_MASK) != 0)
        
        # Add underflow bit to output
        while self._underflow_bits > 0:
          encoded.append((self._upper & MSB_MASK) == 0)
          self._underflow_bits -= 1

      # Resolve underflow problem self._lower = 011..., self._upper = 100...
      # Remove second most significant bit
      elif (self._lower & SMSB_MASK) and not(self._upper & SMSB_MASK):
        self._underflow_bits += 1
        self._lower &= ~(MSB_MASK | SMSB_MASK)
        self._upper |= SMSB_MASK
        
      else:
        return encoded
      # Remove potential negative sign
      # Remove the most significant bit
      self._lower <<= 1
      self._upper <<= 1
      self._upper |= 1

      # Remove potential negative sign
      # 0x11704454 vs 0x1f704454
      self._lower &= SIGN_MASK
      self._upper &= SIGN_MASK


  def append_remain_bits(self, encoded):
    
    # Add self._lower's second MSB to output
    # The MSB bit are already added
    encoded.append((self._lower & SMSB_MASK) != 0)
    
    # Add underflow bits to the output
    self._underflow_bits += 1
    for _ in range(self._underflow_bits):
      encoded.append((self._lower & SMSB_MASK) == 0)
    return encoded

  def encode(self, inp):
    
    # Convert string to stream
    # char level: stream = ['t', 'h', 'e', ' ', 'f', ...]
    # word level: stream =['the', 'fox', 'jumped', ...]
    stream, count = self.streaming(inp)
    encoded = []

    for idx, s in enumerate(stream):
      # print('Trace after append_bits: lower {} upper {}'.format(self._lower/SIGN_MASK, self._upper/SIGN_MASK))
      
      # Compute the current range
      current_range = self._upper - self._lower + 1
      
      # Get context from input for the statistical model
      context = self.get_context(stream[:idx])
      # print('current range {}'.format(current_range/SIGN_MASK))
      # print('Symbol: {}'.format(s))
      
      # Update the new lower bound and upper bound from the distribution
      self._upper = self._lower + int(current_range * self._model.get_upper(s, context)) - 1
      self._lower = self._lower + int(current_range * self._model.get_lower(s, context))
      # print('Trace before append_bits: lower {} upper {}'.format(self._lower/SIGN_MASK, self._upper/SIGN_MASK))
      encoded = self.append_bits(encoded)
    
    encoded = self.append_remain_bits(encoded)
    return encoded, count

  def read_encoded_bits(self, code, inp):
    
    # Similar to append_bits() function
    # print('read_encoded_bits: lower, upper',self._lower, self._upper)
    
    while True:
      if (self._upper ^ ~self._lower) & MSB_MASK:
        # Only shift bit
        pass

      elif (~self._upper & self._lower) & SMSB_MASK:
        # print('under flow ', code, bin(SMSB_MASK))
        self._lower &= ~(MSB_MASK | SMSB_MASK)
        self._upper |= SMSB_MASK
        
        # Remove the 2nd MSB from code
        # Note that self._lower = 01... 
        # and self._upper = 10...
        code ^= SMSB_MASK

      else:
        return code

      # self._lower &= SIGN_MASK
      self._lower <<= 1
      # self._upper &= SIGN_MASK
      self._upper <<= 1
      self._upper |= 1
      
      self._lower &= SIGN_MASK
      self._upper &= SIGN_MASK
      
      code <<= 1
      code &= SIGN_MASK
      if len(inp) > 0:
        bit = inp.pop(0)
        code |= bit
      
  def decode(self, inp, length):
    
    # Initialize the decoder
    self._lower = 0x0
    self._upper = SIGN_MASK
    code = 0
    
    # Read the first PRECISION bits
    for i in range(PRECISION):
      code <<= 1
      if len(inp) > 0:
        bit = inp.pop(0)
        code |= bit
    out_stream = []
    count = 0
    
    while True:
      # Stop decode when reach the original length
      if count == length:
        break
      count += 1
      
      current_range = self._upper - self._lower + 1
      
      # Calculative the cummulative probability range
      prob_range = (code - self._lower + 1) / current_range
      
      # print('prob_range',prob_range)
      # Get context to get current symbol's probability
      context = self.get_context(out_stream)
      # print(context)
      s = self._model.get_symbol(prob_range, context)
      
      # Append new decoded symbol to the output
      out_stream.append(s)
      
      # print('current range {}, code {}'.format(current_range, code))
      self._upper = self._lower + int(current_range * self._model.get_upper(s, context)) - 1
      self._lower = self._lower + int(current_range * self._model.get_lower(s, context))
      
      # print('trace: lower {} upper {}'.format(self._lower, self._upper))
      code = self.read_encoded_bits(code, inp)
    return out_stream