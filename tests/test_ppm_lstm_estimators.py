import pytest
from unittest.mock import Mock

from mlzip.estimators import PPMEstimator, LSTM
from mlzip.arithmetic_coding import ArithmeticCoding

COUNTER_ARRAY = [0, 1, 2, 3, 4, 5, 6]

ABCD = 'abcd' * 1000

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

configs = {
  'context_width': 12,
  'embed_size': 20,
  'hidden_size': 20,
  'num_layers': 2,
  'epochs': 20,
  'lr': 1e-2
}


@pytest.mark.parametrize('mode, expected', testcase_encode_decode)
def test_encode_decode(mode, expected):
  model = LSTM
  estimator = PPMEstimator(mode, configs.get('context_width'), model, configs)
  estimator.fit(LOREM_IPSUM)
  frequency_ac = ArithmeticCoding(estimator)
  encoded, length = frequency_ac.encode(LOREM_IPSUM)
  decoded = frequency_ac.decode(encoded, length)
  assert decoded == expected
