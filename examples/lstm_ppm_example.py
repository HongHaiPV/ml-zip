import sys
from mlzip.arithmetic_coding import ArithmeticCoding
from mlzip.estimators import PPMEstimator, LSTM

tiny_model = {
  'context_width': 32,
  'embed_size': 16,
  'hidden_size': 32,
  'num_layers': 2,
  'epochs': 30,
  'lr': 1e-2
}

if __name__ == '__main__':
  with open('example.txt', 'r') as f:
    text = f.read()
  model_type = LSTM
  estimator = PPMEstimator(stream_type='char',
                           context_width=tiny_model.get('context_width'),
                           model_type=model_type,
                           model_configs=tiny_model)
  org_size = len(text)*8
  print('Original size {} bits.'.format(org_size))
  estimator.fit(text)
  ac = ArithmeticCoding(estimator)
  encoded, length = ac.encode(text)
  decoded = ac.decode(encoded, length)
  assert ''.join(decoded) == text
  encoded_size = len(encoded)
  model_size = sys.getsizeof(estimator.model.state_dict())*8
  print('The file was encoded to {} bits with the parameter size of {} bits'
        '.'.format(encoded_size, model_size))
  print('Compression ratio: {:.2f}.'.format(org_size/(model_size+encoded_size)))
