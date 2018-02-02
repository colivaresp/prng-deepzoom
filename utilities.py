#
# Utility function for selectively loading the content of training or test sample files into a single memory array
#

import numpy as np
from tqdm import tqdm

TRAIN_PATTERN = 'data/DatasetKlogmap/Train/k%d/k%d-parte%d.ser'
TEST_PATTERN = 'data/DatasetKlogmap/Test/k%d/k%d-parte%d.ser'

def load_data(k, path_pattern=TRAIN_PATTERN, indices=[0]):
  data = []
  print('Loading data for k=%d, indices=' % k, indices)
  for i in tqdm(indices):
    with open(path_pattern % (k, k, i), 'r') as f:
      data += [int(i) for i in f.read()]
  return data


#
# Utility functions for plotting summary data exported from TensorBoard as csv files
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def plot_graph_from_tensorboard_csv(filename, 
                                    title, 
                                    xlabel = 'Millions of digits seen', 
                                    xcol = 'Step', 
                                    ycol = 'Value', 
                                    format_string = 'r', 
                                    batch_data_size = 32 * 192,
                                    cross_entropy_to_perplexity = False,
                                    ylim = None,
                                    step_range = None
                                   ):
  df = pd.read_csv(filename)
  if cross_entropy_to_perplexity:
    df[ycol] = np.exp( df[ycol])
  if step_range:
    start, end = step_range
    df = df[start:end]
  if ylim:
    plt.ylim(ylim)
  plt.plot(df[xcol] * batch_data_size, df[ycol], format_string)
  plt.title(title + ' ~ {:.04f}'.format(np.mean(df[ycol][-1000:])) )
  plt.xlabel(xlabel)

def plot_graphs(file_prefix, 
                step_size = 32, 
                batch_size = 192, 
                main_title = None, 
                step_range = None, 
                fit_y_scale_to_data = False,
                marker = ''
               ):
  plt.style.use('ggplot')
  plt.figure(figsize=(16,4))
  plt.tight_layout()
  if main_title:
    plt.suptitle(main_title, size=16)
    plt.subplots_adjust(top=0.82) 
  
  ylim = None
  plt.subplot(1, 3, 1)
  if not(fit_y_scale_to_data):
    ylim = (0,1)
  plot_graph_from_tensorboard_csv(filename = file_prefix + '-tag-summaries_accuracy.csv',
                                  title = 'Accuracy',
                                  format_string = 'r' + marker,
                                  batch_data_size = step_size * batch_size / 1000000, 
                                  ylim = ylim, 
                                  step_range = step_range
                               )

  plt.subplot(1, 3, 2)
  ylim = None
  plot_graph_from_tensorboard_csv(filename = file_prefix + '-tag-summaries_loss.csv',
                                  title = 'Perplexity',
                                  format_string = 'g' + marker,
                                  batch_data_size = step_size * batch_size / 1000000,
                                  cross_entropy_to_perplexity = True,
                                  ylim = ylim, 
                                  step_range = step_range
                               )
  
  plt.subplot(1, 3, 3)
  plt.yscale('linear')
  if not(fit_y_scale_to_data):
    ylim = (0,1)
  plot_graph_from_tensorboard_csv(filename = file_prefix + '-tag-summaries_mean_likelihood.csv',
                                  title = 'Mean Likelihood',
                                  format_string = 'b' + marker,
                                  batch_data_size = step_size * batch_size / 1000000,
                                  ylim = ylim, 
                                  step_range = step_range
                               )

#
# Utility function for estimating empirical frequencies 'signature'
#

import numpy as np
import pandas as pd
from tqdm import trange
tqdm.monitor_interval = 0

def empirical_frequencies(data,
                          chunk_size = 1000,
                          classes = np.arange(10)
                         ):
  data_len = len(data)
  num_chunks = data_len // chunk_size
  assert num_chunks > 0, 'Chunk size greater than data length'

  empirical_frequencies = np.zeros((num_chunks, len(classes)))
  
  print('Calculating empirical frequencies for {} chunks of size {}'.format(num_chunks, chunk_size))
  
  for i in trange(num_chunks):
    chunk = data[i * chunk_size: (i + 1) * chunk_size]
    empirical_frequencies[i] = [chunk.count(x) / chunk_size for x in classes]
    
  mu = np.mean(empirical_frequencies, 0)
  sigma = np.std(empirical_frequencies, 0)
  
  df = pd.DataFrame({'Mean frequency': mu, 'Standard dev': sigma})
  
  return df


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.amax(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)



#
# From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
#
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.style.use('classic')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


