import os
import numpy as np
import urllib.request
from pycoral.utils import edgetpu, dataset
from pycoral.adapters import common, segment
from PIL import Image

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  indices = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((indices >> channel) & 1) << shift
    indices >>= 3

  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

urllib.request.urlretrieve('https://ultralytics.com/images/bus.jpg', 'bus.jpg')
wd = os.getcwd()
model_file = os.path.join(wd, 'model.tflite')
sample_img = os.path.join(wd, 'bus.jpg')

interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

size = common.input_size(interpreter)
image = Image.open(sample_img).convert('RGB').resize(size, Image.ANTIALIAS)

common.set_input(interpreter, image)
interpreter.invoke()
result = segment.get_output(interpreter)

mask_img = Image.fromarray(label_to_color_image(result).astype(np.uint8))
mask_img.save('out.png')
