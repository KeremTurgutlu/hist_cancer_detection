import warnings; warnings.filterwarnings('ignore')
from fastai.script import *
from fastai.distributed import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.models import cadene_models 
sys.path.append("dev/"); from metric_utils import AUC

# learn.load gives error
class OnlySaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self, learn:Learner, monitor:str='val_loss', mode:str='auto', every:str='improvement', name:str='bestmodel'):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every,self.name = every,name
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'
                 
    def jump_to_epoch(self, epoch:int)->None:
        try: 
            self.learn.load(f'{self.name}_{epoch-1}', purge=False)
            print(f"Loaded {self.name}_{epoch-1}")
        except: print(f'Model {self.name}_{epoch-1} not found.')

    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."
        if self.every=="epoch": self.learn.save(f'{self.name}_{epoch}')
        else: #every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                self.best = current
                self.learn.save(f'{self.name}')

@call_parse
def main(gpu:Param("GPU to run on", str)=None, 
         arch_name:Param("Architecture name", str)="densenet201",
         model_suffix:Param("Model name suffix", str)="_non_overlap",
         fold_num:Param("Architecture name", str)=0):
    
    """Distrubuted training of a given experiment.
    Fastest speed is if you run as follows:
        python ../fastai/fastai/launch.py --gpus=3456 ./multi_gpu_training.py --arch_name=resnet18
        --model_suffix=_non_overlap --fold_num=0
    """
    
    # Init
    path = Path('/data/users/turgutluk/histopathologic/')
    MODEL_NAME = f"{arch_name}{model_suffix}"
    epochs = {'stage1':20, 'stage2':30, 'stage3':50}
    print(f"tranining for epochs {epochs}")

    # Load kfold data
#     cv_data = pd.read_pickle(path/'models/cv_non_overlap_data.pkl')
    cv_data = pd.read_pickle(path/'models/cv_data_sz_224.pkl')
    
    # Get model function
    try:arch = getattr(models, arch_name); print(f"got fastai model {arch_name}")
    except:arch = getattr(cadene_models, arch_name); print(f"got cadene model {arch_name}")

    # Create model dir
    fold_num = int(fold_num)
    print(f"Starting Training with model: {MODEL_NAME} and fold: {fold_num}")
    os.makedirs(path/f'models/best_of_{MODEL_NAME}', exist_ok=True)

    # Init distributed
    gpu = setup_distrib(gpu)
    n_gpus = num_distrib()
        
    # Train with kfold data
    print(f"Initialize Learner at fold{fold_num}")
    fold_data = cv_data[fold_num]
    auc = AUC()
    learn_callbacks = [TerminateOnNaNCallback()]  
    learn_callback_fns = [partial(ReduceLROnPlateauCallback, monitor='auc', mode='max', patience=0, factor=0.9),
                         partial(OnlySaveModelCallback, monitor='auc', mode='max', every='improvement',
                                      name=f'best_of_{MODEL_NAME}/fold{fold_num}'),
                         
                         partial(CSVLogger, filename=f'logs/{MODEL_NAME}', append=True)]
    
    learn = cnn_learner(data=fold_data, base_arch=arch, metrics=[accuracy, auc], 
                    lin_ftrs=[1024,1024], ps=[0.8, 0.8, 0.8],
                    callbacks=learn_callbacks,
                    callback_fns=learn_callback_fns)
    # distributed
    learn.to_fp16()
    learn.to_distributed(gpu)
    
    # Stage-1 training
    lr = 3e-3
    max_lr=lr
    print(f"Stage-1 training with lr={max_lr}")
    learn.fit_one_cycle(epochs['stage1'], max_lr=max_lr)

    # Stage-2 training
    learn.freeze_to(1)
    max_lr=lr/10
    print(f"Stage-2 training with lr={max_lr}")
    learn.fit_one_cycle(epochs['stage2'], max_lr=[max_lr/10, max_lr/3, max_lr])

    # Stage-3 training
    learn.unfreeze()
    max_lr=lr/100
    print(f"Stage-3 training with lr={max_lr}")
    learn.fit_one_cycle(epochs['stage3'], max_lr=[max_lr/10, max_lr/3, max_lr])

    # Save Manually - SaveModelCallback gives error
    learn.save(f'best_of_{MODEL_NAME}/fold{fold_num}')
    print(f"Training of fold{fold_num} model is done")
