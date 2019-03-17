# End-to-end train-predict-submit

import warnings
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.models import cadene_models 
warnings.filterwarnings('ignore')
path = Path('../data/histopathologic/')
sys.path.append("dev/"); from metric_utils import AUC, CSVLogger

# get model name from command line
model_name = sys.argv[1]
epochs = {'stage1':30, 'stage2':30, 'stage3':30}
print(f"tranining for epochs {epochs}")

# load kfold data
cv_data = pd.read_pickle(path/'models/cv_data.pkl')

# get model function
try:
    arch = getattr(models, model_name)
    print(f"got fastai model {model_name}")
except:
    arch = getattr(cadene_models, model_name)
    print(f"got cadene model {model_name}")
model_name = arch.__name__ 
print(f"Starting Training with model: {model_name}")

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


# create preds and submissions directory for the model
os.makedirs(path/f'preds/best_of_{model_name}', exist_ok=True)
os.makedirs(path/f"submissions/best_of_{model_name}", exist_ok=True)

# Initialize Learner
print(f"re-initialize Learner to load trained models")
learn = cnn_learner(data=fold_data, base_arch=arch, metrics=[accuracy, auc], callbacks=learn_callbacks,
               callback_fns=learn_callback_fns)

print("Get preds")
for i in range(5):
    # disable TerminateOnNaNCallback for get_preds to work
    learn.callbacks = [cb for cb in learn.callbacks if
                       cb.__class__ == TerminateOnNaNCallback.__class__]

    # load fold model
    load_fold_num = i
    learn.load(f'best_of_{model_name}/fold{load_fold_num}');

    # get preds
    test_preds, _ = learn.get_preds(ds_type=DatasetType.Test)

    # save preds as pickle file
    pd.to_pickle(test_preds, path/f'preds/best_of_{model_name}/fold{load_fold_num}_preds.pkl')
    
print("Get TTA preds")
for i in range(5):
    # disable TerminateOnNaNCallback for get_preds to work
    learn.callbacks = [cb for cb in learn.callbacks if
                       cb.__class__ == TerminateOnNaNCallback.__class__]

    # load fold model
    load_fold_num = i
    learn.load(f'best_of_{model_name}/fold{load_fold_num}');

    # get preds
    tta_preds,_  = learn.TTA(ds_type=DatasetType.Test)

    # save preds as pickle file
    pd.to_pickle(tta_preds, path/f'preds/best_of_{model_name}/fold{load_fold_num}_TTA_preds.pkl')
    

print("Creating submissions")
sample_submission = pd.read_csv(path/'sample_submission.csv')
test_names = [o.name.split('.')[0] for o in learn.data.test_ds.items]
pred_fnames = (path/f'preds/best_of_{model_name}').ls()

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
avg_TTA_path = path/f"submissions/best_of_{model_name}/{model_name}_avg_TTA.csv"
avg_non_TTA_path = path/f"submissions/best_of_{model_name}/{model_name}_avg_non_TTA.csv"
avg_TTA_submission.to_csv(avg_TTA_path, index=False)
avg_non_TTA_submission.to_csv(avg_non_TTA_path, index=False)


# Submit to kaggle
cmd = "kaggle competitions submit -c histopathologic-cancer-detection -f {} -m {}_avg_TTA".format(avg_TTA_path, model_name)
os.system(cmd)
cmd = "kaggle competitions submit -c histopathologic-cancer-detection -f {} -m {}_avg_TTA".format(avg_non_TTA_path, model_name)
os.system(cmd)


