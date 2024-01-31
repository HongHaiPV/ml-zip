# ml-zip: File compression using Machine Learning
A package for data compression using Machine Learning.

### Waring: 🏗️ Under construction 0.0.1 pre-alpha

## Usecase
- You have a large amount of computational resources but a terrible internet bandwidth to send files.
- You want to get started on the [Hutter Prize](http://prize.hutter1.net/).

## Changelog
- 0.0.1:
  - Add Arithmetic Coding (unlimited precisions) method in plein Python using int64 (or int32).
  - Add the following estimators:
    - Fixed Frequency, Adaptive Frequency (implemented using Fenwick tree).
    - LSTM (used PyTorch).

## Usage
```
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

```
## Author
honghaipvu at gmail dot com

## References
- The implementation of infinite precision Arithmetic Coding using integers follows [Mark Nelson's guide](https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html).

## License
[MIT](https://choosealicense.com/licenses/mit/)
