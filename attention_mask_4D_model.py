# End-to-end train-predict-submit

import warnings
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.models import cadene_models 
from transform_utils import center_attn_mask, center_crop, center_crop_and_resize
warnings.filterwarnings('ignore')
path = Path('../data/histopathologic/')
sys.path.append("dev/"); from metric_utils import AUC, CSVLogger

def convert_model_to_4D_input(model):
    "add 1 additional dim to all convolution kernels e.g. (3,64) -> (4,64) (average of all 3)"
    conv1 = model[0][0][0]
    mean_conv_weight = conv1.weight.data.mean(1).unsqueeze(1)
    new_weight = torch.cat([conv1.weight.data, mean_conv_weight], dim=1)
    new_conv = conv2d(conv1.in_channels+1, conv1.out_channels, 
                      conv1.kernel_size, conv1.stride, padding=conv1.padding).cuda()
    model[0][0][0] = new_conv
    return model

# get model name from command line
model_name = sys.argv[1]
epochs = {'stage1':100, 'stage2':100, 'stage3':100}

# load cv data
print("Load cv data")
cv_data = pd.read_pickle(path/'models/cv_attn_mask_data.pkl')

# get model function
try:
    arch = getattr(models, model_name)
    print(f"got fastai model {model_name}")
except:
    arch = getattr(cadene_models, model_name)
    print(f"got cadene model {model_name}")
model_name = arch.__name__ 
print(f"Starting Training with model: {model_name}")


# Define paths
print("Define path for saving model and logging")
MODEL_DIR = path/f'models/best_of_{model_name}_attn'
PREDS_DIR = path/f'preds/best_of_{model_name}_attn'
SUBS_DIR = path/f'submissions/best_of_{model_name}_attn'
LOG_FNAME = f'logs/{model_name}_attn.csv'
os.makedirs(MODEL_DIR, exist_ok=True)
auc = AUC()

# Train with kfold models
print(f"Starting Training with model: {model_name}")

from fastai.vision.learner import cnn_config
for i in range(5):
    fold_num = i
    fold_data = cv_data[i]

    learn_callbacks = [TerminateOnNaNCallback()]
    learn_callback_fns = [partial(EarlyStoppingCallback, monitor='auc', mode='max', patience=2),

                          partial(SaveModelCallback, monitor='auc', mode='max', every='improvement',
                                  name=Path(MODEL_DIR.name)/f'fold{fold_num}'),

                          partial(CSVLogger, filename=LOG_FNAME)]

    learn = cnn_learner(data=fold_data, base_arch=arch, metrics=[accuracy, auc], callbacks=learn_callbacks,
                        callback_fns=learn_callback_fns)

    # change model to 4D input
    convert_model_to_4D_input(learn.model);

    # re-assign new 4d parameters to layer groups and freeze model 
    meta = cnn_config(arch)
    learn.split(meta['split']);
    learn.freeze()

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

# create preds and submissions directory for the model
os.makedirs(PREDS_DIR, exist_ok=True)
os.makedirs(SUBS_DIR, exist_ok=True)

# re-initialize destroyed learner
print(f"Initialize Learner to load trained models")
learn = cnn_learner(data=fold_data, base_arch=arch, metrics=[accuracy, auc], callbacks=learn_callbacks,
               callback_fns=learn_callback_fns)
convert_model_to_4D_input(learn.model);
meta = cnn_config(arch)
learn.split(meta['split']);

# get preds
print(f"Get kfold preds")
for i in range(5):
    # disable TerminateOnNaNCallback for get_preds to work
    learn.callbacks = [cb for cb in learn.callbacks if
                       cb.__class__ == TerminateOnNaNCallback.__class__]

    # load fold model
    load_fold_num = i
    learn.load(Path(MODEL_DIR.name)/f'fold{load_fold_num}')

    # get preds
    test_preds, _ = learn.get_preds(ds_type=DatasetType.Test)

    # save preds as pickle file
    pd.to_pickle(test_preds, PREDS_DIR/f'fold{load_fold_num}_preds.pkl')
    
# get TTA preds
print(f"Get kfold TTA preds")
for i in range(5):
    # disable TerminateOnNaNCallback for get_preds to work
    learn.callbacks = [cb for cb in learn.callbacks if
                       cb.__class__ == TerminateOnNaNCallback.__class__]

    # load fold model
    load_fold_num = i
    learn.load(Path(MODEL_DIR.name)/f'fold{load_fold_num}')

    # get preds
    tta_preds,_  = learn.TTA(ds_type=DatasetType.Test)

    # save preds as pickle file
    pd.to_pickle(tta_preds, PREDS_DIR/f'fold{load_fold_num}_TTA_preds.pkl')
    
# create submission
sample_submission = pd.read_csv(path/'sample_submission.csv')
test_names = [o.name.split('.')[0] for o in learn.data.test_ds.items]

pred_fnames = (PREDS_DIR).ls()
TTA_fnames = [o for o in pred_fnames if "TTA" in str(o)]
non_TTA_fnames = [o for o in pred_fnames if "TTA" not in str(o)]

# average TTA_preds
avg_TTA_labels = np.mean([pd.read_pickle(fn).numpy() for fn in TTA_fnames], axis=0)[:, 1]
avg_non_TTA_labels = np.mean([pd.read_pickle(fn).numpy() for fn in non_TTA_fnames], axis=0)[:, 1]

# create submission file
avg_TTA_submission = sample_submission.copy()
avg_non_TTA_submission = sample_submission.copy()

avg_TTA_submission['label'] = sample_submission['id'].map(dict(zip(test_names, avg_TTA_labels)))
avg_non_TTA_submission['label'] = sample_submission['id'].map(dict(zip(test_names, avg_non_TTA_labels)))

# save submissions
avg_TTA_path = SUBS_DIR/f"{model_name}_avg_TTA.csv"
avg_non_TTA_path = SUBS_DIR/f"{model_name}_avg_non_TTA.csv"
avg_TTA_submission.to_csv(avg_TTA_path, index=False)
avg_non_TTA_submission.to_csv(avg_non_TTA_path, index=False)

# submit to Kaggle
print("Submitting to Kaggle")
tta_cmd = "kaggle competitions submit -c histopathologic-cancer-detection -f {} -m {}_avg_TTA".format(avg_TTA_path, model_name)
non_tta_cmd = "kaggle competitions submit -c histopathologic-cancer-detection -f {} -m {}_avg_TTA".format(avg_non_TTA_path, model_name)
os.system(tta_cmd); os.system(non_tta_cmd)
