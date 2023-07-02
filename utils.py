import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import rasterio

def plot_confusion_matrix(cm=None,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          path=None,
                          name=None):
    target_names = ['Bareland',
                    'Fruits',
                    'Grass',
                    'Guizota',
                    'Lupine',
                    'Maize',
                    'Millet',
                    'Others',
                    'Pepper',
                    'Teff',
                    'Vegetables'
                   ]
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')    # Blues
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]+0.0001
    cm = np.round(cm,3)
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="pink" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="pink" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    fname = f'{path}/{name}.png'
    plt.savefig(fname=fname, dpi=350, facecolor='auto', edgecolor='auto', bbox_inches='tight', pad_inches=0.1)
    plt.show()
    


def rasterize(temp_rst=None, array=None, path=None, name=None):
    if temp_rst is not None:
        tempelate = rasterio.open(temp_rst).meta
        tempelate.update({'dtype':np.uint8,
                         'nodata':255,
                         'count': 1})
        array = np.nan_to_num(array.reshape(3094, 4408), 255).astype(np.uint8)

        with rasterio.open(f'{path}/{name}.tif', 'w', **tempelate) as dest:
            dest.write(array, 1)
    else:
        raise ValueError('Proper raster tempelate not provided')
