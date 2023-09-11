from typing import Iterable


def rolling_window_context(stream: Iterable, context_width: int, padding: str,
                           index: int):
  """
  Extract context from the stream giving the window size.
  The padding is used when the index is smaller than the window size - 1.

  Args:
    stream (Iterable): Iterable containing the stream data.
    context_width (int): The size of the rolling window.
    padding (str): Special character used to denote the boundary.
    index (int): The current end of the window.

  Returns:
    List: A window of data.
  """

  if index < context_width:
    return [padding] * (context_width - index) + stream[:index]

  return stream[index - context_width:index]


def get_stream_text(input: Iterable, mode: str):
  """
  Turn inputs into stream of symbols.

  Args:
    input (Iterable): The input text.
    mode (str): The smallest streaming elements.

  Returns:
    List[str]: A list of symbols, either characters or words.
  """

  # Character model
  if mode == 'char':
    return list(input), len(input)

  # Word model
  elif mode == 'word':
    return input.split(' '), len(input.split(' '))

  else:
    return None


def text2bits(input_string):
  return list(map(bin, bytearray(input_string)))
