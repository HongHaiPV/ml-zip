from typing import Iterable, Any, IO

import sys
import time

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


def progressbar(it: Iterable[Any],
                prefix: str = '',
                size: int = 60,
                out: IO = sys.stdout):
  """
  Print out progress bar.

  Args:
    it (Iterable): An iterable to loop through and print progress.
    prefix (str): The string before the bar, default: empty.
    size (int): The size of the bar, default: 60.
    out (IO): Output destination, default: sys.stdout.

  Returns:
    None
  """
  count = len(it)
  start = time.time()

  def show(j):
    x = int(size * j / count)
    if j != 0:
      remaining = ((time.time() - start) / j) * (count - j)
    else:
      remaining = 0
    mins, secs = divmod(remaining, 60)
    time_str = f"{int(mins):02}:{secs:05.2f}"
    print(f"{prefix}[{u'█' * x}{('.' * (size - x))}] {j}/{count}\
          Est wait {time_str}", end='\r',
          file=out, flush=True)

  show(0)
  for i, item in enumerate(it):
    yield item
    show(i + 1)
  print("\n", flush=True, file=out)
