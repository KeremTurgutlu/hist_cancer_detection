import warnings; warnings.filterwarnings('ignore')
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.models import cadene_models 
sys.path.append("dev/")
from metric_utils import AUC, CSVLogger
from core_utils import *


@call_parse
def main(arch_name:Param("Architecture name", str)="densenet201",
         model_suffix:Param("Model name suffix", str)="_non_overlap",
         data_file=(path/'models/cv_data.pkl'), gpu:Param("GPU id", str)="0"):
    
    """Distrubuted training of a given experiment.
    Fastest speed is if you run as follows:
        python ../fastai/fastai/launch.py --gpus=3456 ./multi_gpu_training.py --arch_name=resnet18
        --model_suffix=_non_overlap --fold_num=0
    """

    # Init
    gpu = int(gpu)
    path = Path('/data/users/turgutluk/histopathologic/')
    MODEL_NAME = f"{arch_name}{model_suffix}"
    epochs = {'stage1':20, 'stage2':30, 'stage3':50}
    print(f"tranining for epochs {epochs}")
    
    # load kfold data
    cv_data = pd.read_pickle(data_file)

    # get model function
    arch = get_arch_by_name(arch_name)

    print(f"Starting Training... {MODEL_NAME}")

    # create model dir
    os.makedirs(path/f'models/best_of_{model_name}', exist_ok=True)

    # Train with kfold models
    for i in range(len(cv_data)):
        fold_num = i
        fold_data = cv_data[i]

        # Initialize Learner
        print(f"Initialize Learner at fold{fold_num}")
        auc = AUC()
        learn_callbacks = [TerminateOnNaNCallback()]
        learn_callback_fns = [partial(EarlyStoppingCallback, monitor='auc', mode='max', patience=2),

                              partial(SaveModelCallback, monitor='auc', mode='max', every='improvement',
                                      name=f'best_of_{model_name}/fold{fold_num}'),

                              partial(CSVLogger, filename=f'logs/{model_name}')]

        learn = cnn_learner(data=fold_data, base_arch=arch, metrics=[accuracy, auc], callbacks=learn_callbacks,
                       callback_fns=learn_callback_fns)

        print(f"Number of layer groups for {model_name}: {len(learn.layer_groups)}")

        # Stage-1 training
        learn.lr_find()
        try:
            learn.recorder.plot(suggestion=True, k=5)
        except: 
             learn.recorder.plot(suggestion=True)
        max_lr = learn.recorder.min_grad_lr
        print(f"Stage-1 training with lr={max_lr}")
        learn.fit_one_cycle(epochs['stage1'], max_lr=max_lr)

        # Stage-2 training
        learn.freeze_to(1)
        learn.lr_find()
        try:
            learn.recorder.plot(suggestion=True, k=5)
        except: 
             learn.recorder.plot(suggestion=True)
        max_lr = learn.recorder.min_grad_lr
        print(f"Stage-2 training with lr={max_lr}")
        learn.fit_one_cycle(epochs['stage2'], max_lr=[max_lr/10, max_lr/3, max_lr])

        # Stage-3 training
        learn.unfreeze()
        learn.lr_find()
        try:
            learn.recorder.plot(suggestion=True, k=5)
        except: 
             learn.recorder.plot(suggestion=True)
        max_lr = learn.recorder.min_grad_lr
        print(f"Stage-3 training with lr={max_lr}")
        learn.fit_one_cycle(epochs['stage3'], max_lr=[max_lr/10, max_lr/3, max_lr])

        print(f"Training of fold{fold_num} model is done...destroying learner")
        learn.destroy()


