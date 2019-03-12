from fastai.vision import *
from sklearn.metrics import roc_auc_score

class AUC(Callback):
    "AUC score"
    def __init__(self):
        pass
    
    def on_epoch_begin(self, **kwargs): 
        self.outputs = []
        self.targets = []

    def on_batch_end(self, last_output, last_target, **kwargs):
        "expects binary output with data.c=2 "
        self.outputs += list(to_np(last_output)[:, 1])
        self.targets += list(to_np(last_target))

    def on_epoch_end(self, last_metrics, **kwargs): 
        return {'last_metrics': last_metrics + [roc_auc_score(self.targets, self.outputs)]}
    

class CSVLogger(LearnerCallback):
    "A `LearnerCallback` that saves history of metrics while training `learn` into CSV `filename`."
    def __init__(self, learn:Learner, filename: str = 'history', append: bool = False): 
        super().__init__(learn)
        self.filename,self.path, self.append = filename,self.learn.path/f'{filename}.csv', append

    def read_logged_file(self):  
        "Read the content of saved file"
        return pd.read_csv(self.path)

    def on_train_begin(self, **kwargs: Any) -> None:
        "Prepare file with metric names."
        self.path.parent.mkdir(parents=True, exist_ok=True)      
        self.file = self.path.open('w') if self.append else self.path.open('a')
        self.file.write(','.join(self.learn.recorder.names) + '\n')
        
    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        "Add a line with `epoch` number, `smooth_loss` and `last_metrics`."
        last_metrics = ifnone(last_metrics, [])
        stats = [str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                 for name, stat in zip(self.learn.recorder.names, [epoch, smooth_loss] + last_metrics)]
        str_stats = ','.join(stats)
        self.file.write(str_stats + '\n')

    def on_train_end(self, **kwargs: Any) -> None:  
        "Close the file."
        self.file.close()