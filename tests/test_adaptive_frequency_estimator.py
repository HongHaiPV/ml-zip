import pytest
from unittest.mock import Mock

from arithmetic_coding.arithmetic_coding import ArithmeticCoding
from arithmetic_coding.estimators import AdaptiveFrequencyEstimator


ABCD = 'abcd'*1000

LOREM_IPSUM = ("<p>Lorem ipsum dolor sit amet. Ut mollitia dolores vel harum"
               "galisum galisum aut obcaecati numquam ut architecto dolorem! "
               "Non minima similique ut nesciunt molestiae ex nemo perferendis"
               " sit voluptate blanditiis id illum magnam est voluptates"
               " praesentium. Cum consequatur facere aut veritatis quibusdam"
               " eum aliquam voluptates qui autem omnis cum tenetur culpa sed"
               " sunt consequuntur vel soluta galisum! </p>"
              )

testcase_encode_decode = [
              ('char', list(LOREM_IPSUM)),
              (('word'), LOREM_IPSUM.split(' '))
            ]

@pytest.fixture
def frequency_table(request):
  frequency_table = AdaptiveFrequencyEstimator('char', 12)
  frequency_table.fit(ABCD)
  return frequency_table

def test_get_context(frequency_table):
  expected_cdf = [0, 0.25, 0.5, 0.75, 1.0]
  context = frequency_table.get_context(ABCD, 42)
  assert context == 'cdabcdabcdab'
  assert all(abs(x1-x2) < 1e-6 for x1, x2 in zip(frequency_table.cdf, expected_cdf))

def test_fit(frequency_table):
  expected_cdf = [0, 0.25, 0.5, 0.75, 1.0]
  assert all(abs(x1-x2) < 1e-6 for x1, x2 in zip(frequency_table.cdf, expected_cdf))


def test_get_upper(frequency_table):
  assert frequency_table.get_upper('a') == pytest.approx(.25)
  assert frequency_table.get_upper('d') == 1.0


def test_get_lower(frequency_table):
  assert frequency_table.get_lower('a') == 0
  assert frequency_table.get_lower('d') == pytest.approx(0.75)


def test_get_symbol(frequency_table):
  assert frequency_table.get_symbol(0.1) == 'a'
  assert frequency_table.get_symbol(0.0) == 'a'
  assert frequency_table.get_symbol(0.75999) == 'd'
  assert frequency_table.get_symbol(0.99999) == 'd'


@pytest.mark.parametrize('mode, expected', testcase_encode_decode)
def test_encode_decode(mode, expected):
  frequency_table = AdaptiveFrequencyEstimator(mode, 12)
  frequency_table.fit(LOREM_IPSUM)
  frequency_ac = ArithmeticCoding(frequency_table)
  encoded, length = frequency_ac.encode(LOREM_IPSUM)
  decoded = frequency_ac.decode(encoded, length)
  assert decoded == expected