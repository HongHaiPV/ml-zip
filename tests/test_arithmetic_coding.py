import pytest
from unittest.mock import Mock

from arithmetic_coding.arithmetic_coding import *

LOWER_TEST = 0xabc
UPPER_TEST = 0xdef
UNDER_BITS = 1
CODE_LENGT = 3
INPUT_CODE = [False, True, False, True]

ABCD = "abcddacb"
L_CDF = {0:0.0, 1:0.25, 2: 0.5, 3:0.75, 4:1.0}
S_IDX = {'a':0, 'b':1, 'c':2, 'd':3}
R_IDX = {0:'a', 1:'b', 2:'c', 3:'d'}


def dynamic_get_lower(symbol, context):
  return L_CDF[S_IDX[symbol]]

def dynamic_get_upper(symbol, context):
  return L_CDF[S_IDX[symbol] + 1]

def dynamic_get_symbol(prob, context):
  for idx in L_CDF:
    if idx == len(L_CDF)-1:
      return R_IDX[idx]
    if L_CDF[idx] <= prob and L_CDF[idx+1] > prob:
      return R_IDX[idx]

def dynamic_get_stream(input):
  return list(input), len(input)

@pytest.fixture
def mock_model():
  mock = Mock()
  mock.get_context.return_value = None
  mock.get_stream.side_effect = dynamic_get_stream
  mock.get_lower.side_effect = dynamic_get_lower
  mock.get_upper.side_effect = dynamic_get_upper
  mock.get_symbol.side_effect = dynamic_get_symbol
  return mock

@pytest.fixture
def ac(mock_model):
  return ArithmeticCoding(mock_model)

@pytest.fixture
def encoder_state():
  return EncoderState(lower = LOWER_TEST,
                      upper = UPPER_TEST,
                      underflow_bits = UNDER_BITS)

@pytest.fixture
def decoder_state():
  return DecoderState(lower = LOWER_TEST,
                      upper = UPPER_TEST,
                      code_length = CODE_LENGT)

def test_append_bits(ac, encoder_state):
  encoded_chunk = ac.append_bits(encoder_state)
  assert encoded_chunk == [False, True, False, False, False, False, False, 
                          False, False, False, False, False, False, False,
                          False, False, False, False, False, False, False, True]
  assert encoder_state.lower == 0x2f000000
  assert encoder_state.upper == 0xfbffffff
  assert encoder_state.underflow_bits == 1
  
def test_append_remain_bits(ac, encoder_state):
  encoded_chunk = ac.append_remain_bits(encoder_state)

  assert encoded_chunk == [False, True, True]
  assert encoder_state.lower == 0xabc
  assert encoder_state.upper == 0xdef
  assert encoder_state.underflow_bits == 2

def test_parse_encoded_bits(ac, decoder_state):
  ac.parse_encoded_bits(decoder_state, INPUT_CODE)
  
  assert decoder_state.lower == 0x2f000000
  assert decoder_state.upper == 0xfbffffff
  assert decoder_state.code == 0x80100000
  assert decoder_state.index == CODE_LENGT

def test_encode(ac):
  encoded, length = ac.encode(ABCD)
  assert encoded == [False, False, False, True, True, False, True, True, True,
                     True, False, False, True, False, False, True, False, True]
  assert length == len(ABCD)

def test_decode(ac):
  encoded, length = ac.encode(ABCD)
  output = ac.decode(encoded, length)
  assert ''.join(output) == ABCD