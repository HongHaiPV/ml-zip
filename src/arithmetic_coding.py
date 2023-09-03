from dataclasses import dataclass, field

PRECISION = 32
MSB_MASK = 1 << PRECISION - 1
SMSB_MASK = 1 << PRECISION - 2
SIGN_MASK = (1 << PRECISION) - 1


@dataclass
class ScaledRangesState:
  """
  Base state for probability range in Arithmetic Encoding.

  Attributes:
    lower (int): The lower probability range, represented by a sequence of bits.
      Default is infinite 0 bits, represented by 0x0 as the first PRECISION
      bits.

    upper (int): The upper probability range, represented by a sequence of
      bits. Default is infinite 1 bits, represented by 0x0 as the first
      PRECISION bits.
  """

  lower: int = field(default=0x0)
  upper: int = field(default=SIGN_MASK)

  def get_range(self):
    """
    Estimate current probability range.

    Returns:
      The current range, represented by an integer.
    """

    return self.upper - self.lower + 1 


@dataclass
class EncoderState(ScaledRangesState):
  """
  State for the encoder in Arithmetic Encoding.

  Attributes:
    underflow_bits (int): number of mismatched bits when both lower and upper
      range close to 0.5
  """

  underflow_bits: int = field(default=0)


@dataclass
class DecoderState(ScaledRangesState):
  """
  State for the decoder in Arithmetic Encoding.
  
  Attributes:
    code_length (int): The number of bits of the encoded message.
    code (int): The window of PRECISION (default to 32) bits that we're
      currently processing.
    index (int): The current position of the least significant bit in the code.
  """

  code_length: int = field(default=0)
  code: int = field(default=0)
  index: int = field(default=0)


class ArithmeticCoding:
  """
  A class for encoding and decoding data using the Arithmetic Encoding algorithm.

  When implemented as infinite precision using integers, the lower and upper
  can converge around 0.5 where lower takes the form of 0b011111..., upper
  takes the form of 0b100000... and the MSB bits can never match. We can
  avoid this problem by removing the second MSB bits from both lower and
  upper when they are 0b01... and 0b10... respectively and keep track of the
  number of occurrences. Eventually, when the MSB bits of lower and upper
  match, we can know that where lower and upper is now either smaller or larger
  than 0.5. If it's smaller, both lower and upper should take the form of 0b01
  followed by finite number of digits and vice versa. That means, when the MSB
  matching happens, we can reconstruct the discarded underflow bits by
  reversing the matched MSB bit.
  """

  def __init__(self, estimator):
    self.estimator = estimator
    self.get_context = estimator.get_context
    self.get_stream = estimator.get_stream

  @staticmethod
  def append_bits(state):
    """
    Continuously convert the input stream into the range of probabilities and 
    return the identical MSB bits.

    Args:
      state: The lower, upper range of probabilities and underflow_bits for
        dealing with the underflow condition

    Returns:
      encoded_chunk: The encoded MSB bits.
    """

    encoded_chunk = []
    while True:
      if (state.upper & MSB_MASK) == (state.lower & MSB_MASK):
        # Add the matched MSB to the output
        encoded_chunk.append((state.lower & MSB_MASK) != 0)

        # Add underflow bits to the output.
        # It's the reverse of the matched MSB.
        while state.underflow_bits > 0:
          encoded_chunk.append((state.lower & MSB_MASK) == 0)
          state.underflow_bits -= 1

      # Resolve underflow problem self.lower = 0b01X, self.upper = 0b10Y
      # where X and Y is any combination of 0s and 1s.
      # Remove second most significant bit as follows:
      # lower = 0b01X; lower = lower & 0b00... = 0b00X
      # upper = 0b10Y; upper = upper | 0b01... = 0b11Y
      # After that, the MSB bits of both upper and lower will be removed by
      # left shifting.
      elif (state.lower & SMSB_MASK) and not(state.upper & SMSB_MASK):
        state.underflow_bits += 1
        state.lower &= ~(MSB_MASK | SMSB_MASK)
        state.upper |= SMSB_MASK
        
      else:
        return encoded_chunk
      
      # Remove the most significant bit
      state.lower <<= 1
      state.upper <<= 1
      state.upper |= 1

      # Remove potential negative sign causes by Python's large int
      # implementation i.e. 0x11704454 vs 0x1f704454
      state.lower &= SIGN_MASK
      state.upper &= SIGN_MASK

  @staticmethod
  def append_remain_bits(state):
    """
    Attach the remaining bits from the state variable to the encoded bits.
    At this stage, we need to make sure lower <= output < upper.
    Since the current (lower, upper) can be either (0b00..., 0b11...),
    (0b00..., 0b10...), or (0b01..., 0b11...) the output only need to take
    the second MSB of the lower, plus any underflow_bits and another one to
    satisfy this condition.

    Args:
      state: The state of current probabilities range.

    Returns:
      encoded_chunk: The remained bits from the probabilities range.
    """

    # Get the second MSB bit of the lower.
    encoded_chunk = [(state.lower & SMSB_MASK) != 0]

    state.underflow_bits += 1
    for _ in range(state.underflow_bits):
      encoded_chunk.append((state.lower & SMSB_MASK) == 0)
    return encoded_chunk

  def encode(self, input):
    """
    Encode the input data using arithmetic coding.

    Args:
      input: An iterable object that needs to be encoded.

    Returns:
      encoded: Encoded data as a list of bits.
      length: Length of the original data stream.
    """
    state = EncoderState()
    self.estimator.mode('encode')
    stream, length = self.get_stream(input)
    encoded = []

    for idx, s in enumerate(stream):
      current_range = state.get_range()
      
      # Get context from input for the statistical model
      context = self.get_context(stream, idx)
      
      # Update the new lower bound and upper bound from the distribution
      state.upper = state.lower + int(current_range * self.estimator.get_upper(s)) - 1
      state.lower = state.lower + int(current_range * self.estimator.get_lower(s))
      encoded += self.append_bits(state)
    
    encoded += self.append_remain_bits(state)
    return encoded, length

  def parse_encoded_bits(self, state, input):
    """
    Digest the encoded bits and update the decoder state accordingly.
    
    The decoder state contains lower and upper probability range, the current
    index of the input encoded bits and the current working windows on
    these bits: code.
    
    Args:
      state: The state variable of the current probability range.
      input: The encoded bits.

    Return:
      None 
    """
    
    while True:
      if (state.upper & MSB_MASK) == (state.lower & MSB_MASK):
        # Only shift bits
        pass

      elif (state.lower & SMSB_MASK) and not(state.upper & SMSB_MASK):
        state.lower &= ~(MSB_MASK | SMSB_MASK)
        state.upper |= SMSB_MASK
        
        # Remove the 2nd MSB from code
        # Note that state.lower = 01... 
        # and state.upper = 10...
        state.code ^= SMSB_MASK

      else:
        return

      # Shift bits
      state.lower <<= 1
      state.upper <<= 1
      state.upper |= 1
      
      state.lower &= SIGN_MASK
      state.upper &= SIGN_MASK
      
      state.code <<= 1
      state.code &= SIGN_MASK
      if state.index < state.code_length:
        state.code |= input[state.index]
        state.index += 1
      
  def decode(self, input, length):
    """
    Turn stream of encoded bits into decoded symbols.

    Args:
      input: The encoded bits.
      length: The length of original data.

    Return:
      The decoded stream.

    """
    state = DecoderState(code_length=len(input))
    self.estimator.mode('decode')
    for i in range(PRECISION):
      state.code <<= 1
      if state.index < state.code_length:
        state.code |= input[state.index]
        state.index += 1

    out_stream = []
    count = 0
    
    while True:
      # Stop decode when reach the original length
      if count == length:
        break
      
      current_range = state.get_range()
      
      # Calculative the cummulative probability range
      prob_range = (state.code - state.lower + 1) / current_range
      
      # Get context to get current symbol's probability
      self.get_context(out_stream, count)
      symbol = self.estimator.get_symbol(prob_range)
      out_stream.append(symbol)
      count += 1
      
      state.upper = state.lower + int(current_range * self.estimator.get_upper
        (symbol)) - 1
      state.lower = state.lower + int(current_range * self.estimator.get_lower
        (symbol))
      
      self.parse_encoded_bits(state, input)

    return out_stream