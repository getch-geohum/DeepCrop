import numpy as np
import torch
import argparse
from plot_confusion_report import *
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from scipy import stats as st


class Decision_fuse:
    def __init__(self, root, model_name):
        self.root=root
        self.model_name=model_name
        self.rep_fold=f'{root}/LOGS/{model_name}'

    def decision_fusion(self):
        ind = np.load(f'{self.root}/samples/test_index.npy')
        y = np.load(f'{self.root}/samples/terrasarx/train_test/y_test.npy', allow_pickle=True).ravel() # basically equals with terrasarx lengths
        pred_tsx = np.load(f'{self.root}/LOGS/{self.model_name}/{self.model_name}_terrasarx_test.npy').ravel()
        pred_plnt = np.load(f'{self.root}/LOGS/{self.model_name}/{self.model_name}_planet_test.npy').ravel()[ind]
        pred_sent = np.load(f'{self.root}/LOGS/{self.model_name}/{self.model_name}_sentinel_test.npy').ravel()[ind]
        assert pred_tsx.shape == pred_plnt.shape == pred_sent.shape, 'Array shapes are not the same!'

        stack = np.dstack((pred_sent, pred_plnt, pred_tsx))[0]
        print(stack.shape)
        dfused = st.mode(stack, axis=1)
        dfused = dfused.mode.ravel()

        assert y.shape == dfused.shape, f'shapes of refrenc: {y.shape} and predicted: {dfused.shape} are not the same'

        f1 = f1_score(y_true=y, y_pred=dfused, average='weighted')

        ac = accuracy_score(y, dfused)
        text = open(f'{self.rep_fold}/fused_test_report_deci.txt','w')
        text.write(f"Overall accuracy: {ac}\n")
        text.write(f"Micro F-1 score: {f1}\n")

        print('=======================================')
        print(f'Overall test acuracy: {ac}')
        print(f'Overall f-1 score: {f1}')
        print('=======================================')

        cm_matrics = confusion_matrix(y, dfused, normalize=None)
        names = ['Guizota','Maize','Millet','Others','Pepper','Teff']
        plot_confusion_matrix(cm=cm_matrics,
                              title=f'{self.model_name}_deci_fuse',
                              cmap=None,
                              normalize=True,
                              path=self.rep_fold,
                              target_names=names,
                              fname=f'{self.model_name}_fused_deci',
                              save=True)
        np.save(f'{self.rep_fold}/{self.model_name}_fused_test_deci.npy',dfused)
    

def argumentParser():
    parser = argparse.ArgumentParser(description='Runs and tests crop type mapping using Inception time')
    parser.add_argument('--root', help='Root folder that contained all tensors folders', type=str, default='D:/EO_Africa')
    parser.add_argument('--model', help='Model name for optimization and test', default='Transformer', type=str, required=False)
    args = parser.parse_args()
    return args
    
    
if __name__ == "__main__":
    args = argumentParser()
    fuser = Decision_fuse(root=args.root, model_name=args.model)
    fuser.decision_fusion()