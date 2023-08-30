import pytest
from unittest.mock import Mock

from arithmetic_coding.arithmetic_coding import ArithmeticCoding
from arithmetic_coding.estimator import FrequencyEstimator

LOREM_IPSUM = ("<p>Lorem ipsum dolor sit amet. Ut mollitia dolores vel harum"
               "galisum galisum aut obcaecati numquam ut architecto dolorem! "
               "Non minima similique ut nesciunt molestiae ex nemo perferendis"
               " sit voluptate blanditiis id illum magnam est voluptates"
               " praesentium. Cum consequatur facere aut veritatis quibusdam"
               " eum aliquam voluptates qui autem omnis cum tenetur culpa sed"
               " sunt consequuntur vel soluta galisum! </p>"
              )

@pytest.fixture
def char_frequency_estimator():
  return FrequencyEstimator(LOREM_IPSUM, 'char')

@pytest.fixture
def word_frequency_estimator():
  return FrequencyEstimator(LOREM_IPSUM, 'word')

@pytest.fixture
def char_ac(char_frequency_estimator):
  return ArithmeticCoding(char_frequency_estimator)

@pytest.fixture
def word_ac(word_frequency_estimator):
  return ArithmeticCoding(word_frequency_estimator)

def test_frequency_char(char_ac):
  encoded, length = char_ac.encode(LOREM_IPSUM)
  decoded = char_ac.decode(encoded, length)
  assert decoded == list(LOREM_IPSUM)

def test_frequency_word(word_ac):
  encoded, length = word_ac.encode(LOREM_IPSUM)
  decoded = word_ac.decode(encoded, length)
  assert decoded == LOREM_IPSUM.split()