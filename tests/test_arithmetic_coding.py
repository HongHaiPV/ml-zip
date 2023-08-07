import pytest
from unittest.mock import Mock

from arithmetic_coding.arithmetic_coding import *

LOWER_TEST = 0xabc
UPPER_TEST = 0xdef
UNDER_BITS = 1
CODE_LENGT = 3
INPUT_CODE = [False, True, False, True]

@pytest.fixture
def mock_model():
  mock = Mock()
  mock.get_stream.return_value = None
  mock.get_context.return_value = None
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

def encode():
  pass

def decode():
  pass