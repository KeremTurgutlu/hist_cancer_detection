import warnings; warnings.filterwarnings('ignore')
from fastai.script import *
from fastai.distributed import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.models import cadene_models 
sys.path.append("dev/"); from metric_utils import AUC


@call_parse
def main(gpu:Param("GPU to run on", str)=None, arch_name:Param("Architecture name", str)="densenet201",
         model_suffix:Param("Model name suffix", str)="_non_overlap"):
    """Distrubuted training of a given experiment.
    Fastest speed is if you run as follows:
        python ../fastai/fastai/launch.py --gpus=3456 ./multi_gpu_training.py --arch_name=resnet18 --model_suffix=_non_overlap
    """
    
    # Init
    path = Path('/data/users/turgutluk/histopathologic/')
    MODEL_NAME = f"{arch_name}{model_suffix}"
    epochs = {'stage1':30, 'stage2':50, 'stage3':100}
    print(f"tranining for epochs {epochs}")

    # Load kfold data
    cv_data = pd.read_pickle(path/'models/cv_non_overlap_data.pkl')

    # Get model function
    try:arch = getattr(models, arch_name); print(f"got fastai model {arch_name}")
    except:arch = getattr(cadene_models, arch_name); print(f"got cadene model {arch_name}")

    # Create model dir
    print(f"Starting Training with model: {MODEL_NAME}")
    os.makedirs(path/f'models/best_of_{MODEL_NAME}', exist_ok=True)

    # Init distributed
    gpu = setup_distrib(gpu)
    n_gpus = num_distrib()
        
    # Train with kfold data
    for fold_num in range(len(cv_data)):    
        print(f"Initialize Learner at fold{fold_num}")
        fold_data = cv_data[fold_num]
        auc = AUC()
        learn_callbacks = [TerminateOnNaNCallback()]
        learn_callback_fns = [partial(EarlyStoppingCallback, monitor='auc', mode='max', patience=3),
#                               partial(SaveModelCallback, monitor='auc', mode='max', every='improvement',
#                                       name=f'best_of_{MODEL_NAME}/fold{fold_num}'),
                              partial(ReduceLROnPlateauCallback, monitor='auc', mode='max', patience=0, factor=0.9),
                              partial(CSVLogger, filename=f'logs/{MODEL_NAME}', append=True)]

        learn = cnn_learner(data=fold_data, base_arch=arch, metrics=[accuracy, auc], 
                        lin_ftrs=[1024,1024], ps=[0.7, 0.7, 0.7],
                        callbacks=learn_callbacks,
                        callback_fns=learn_callback_fns)
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
        
        print(f"Training of fold{fold_num} model is done...destroying learner")
        learn.destroy()

        
    # Create preds and submissions directory for the model
    os.makedirs(path/f'preds/best_of_{MODEL_NAME}', exist_ok=True)
    os.makedirs(path/f"submissions/best_of_{MODEL_NAME}", exist_ok=True)

    # Initialize Learner
    print(f"re-initialize Learner to load trained models")
    learn = cnn_learner(data=fold_data, base_arch=arch, metrics=[accuracy, auc], callbacks=learn_callbacks,
                   callback_fns=learn_callback_fns)

    print("Get preds")
    for load_fold_num in range(5):
        # Disable TerminateOnNaNCallback for get_preds to work
        learn.callbacks = [cb for cb in learn.callbacks if
                           cb.__class__ == TerminateOnNaNCallback.__class__]

        # Load fold model
        learn.load(f'best_of_{MODEL_NAME}/fold{load_fold_num}');

        # Get preds
        test_preds, _ = learn.get_preds(ds_type=DatasetType.Test)

        # Save preds as pickle file
        pd.to_pickle(test_preds, path/f'preds/best_of_{MODEL_NAME}/fold{load_fold_num}_preds.pkl')

    print("Get TTA preds")
    for load_fold_num in range(5):
        # Disable TerminateOnNaNCallback for get_preds to work
        learn.callbacks = [cb for cb in learn.callbacks if
                           cb.__class__ == TerminateOnNaNCallback.__class__]

        # Load fold model
        learn.load(f'best_of_{MODEL_NAME}/fold{load_fold_num}');

        # Get preds
        tta_preds,_  = learn.TTA(ds_type=DatasetType.Test)

        # Save preds as pickle file
        pd.to_pickle(tta_preds, path/f'preds/best_of_{MODEL_NAME}/fold{load_fold_num}_TTA_preds.pkl')


    print("Creating submissions")
    sample_submission = pd.read_csv(path/'sample_submission.csv')
    test_names = [o.name.split('.')[0] for o in learn.data.test_ds.items]
    pred_fnames = (path/f'preds/best_of_{MODEL_NAME}').ls()

    TTA_fnames = [o for o in pred_fnames if "TTA" in str(o)]
    non_TTA_fnames = [o for o in pred_fnames if "TTA" not in str(o)]

    # Average TTA_preds
    avg_TTA_labels = np.mean([pd.read_pickle(fn).numpy() for fn in TTA_fnames], axis=0)[:, 1]
    avg_non_TTA_labels = np.mean([pd.read_pickle(fn).numpy() for fn in non_TTA_fnames], axis=0)[:, 1]

    # Create submission file
    avg_TTA_submission = sample_submission.copy()
    avg_non_TTA_submission = sample_submission.copy()

    avg_TTA_submission['label'] = sample_submission['id'].map(dict(zip(test_names, avg_TTA_labels)))
    avg_non_TTA_submission['label'] = sample_submission['id'].map(dict(zip(test_names, avg_non_TTA_labels)))


    # Save submissions
    avg_TTA_path = path/f"submissions/best_of_{MODEL_NAME}/{MODEL_NAME}_avg_TTA.csv"
    avg_non_TTA_path = path/f"submissions/best_of_{MODEL_NAME}/{MODEL_NAME}_avg_non_TTA.csv"
    avg_TTA_submission.to_csv(avg_TTA_path, index=False)
    avg_non_TTA_submission.to_csv(avg_non_TTA_path, index=False)


    # Submit to kaggle
    cmd = "kaggle competitions submit -c histopathologic-cancer-detection -f {} -m {}_avg_TTA".format(avg_TTA_path, MODEL_NAME)
    os.system(cmd)
    cmd = "kaggle competitions submit -c histopathologic-cancer-detection -f {} -m {}_avg_TTA".format(avg_non_TTA_path, MODEL_NAME)
    os.system(cmd)


