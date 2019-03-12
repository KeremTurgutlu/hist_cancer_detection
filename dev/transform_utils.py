from fastai.vision import *
from fastai.vision.transform import _crop_default

def _rand_center_crop(x, a=40, b=96):
    "x: tensor, center crop an image creating a patch randomly between crop size (a,b)"
    crop_sz = np.random.randint(a,b)
    return _crop_default(x, crop_sz)

def _rand_center_crop_and_resize(x, a=40, b=96, targ_sz=(96, 96)):
    "x: tensor, _rand_center_crop and resize to targ_sz at the end"
    crop_sz = np.random.randint(a,b)
    x = _crop_default(x, crop_sz)
    return Image(x).resize((x.shape[0],) + targ_sz).data

def _create_center_mask(x, orig_sz=96, targ_sz=32):
    cnt_mask = torch.zeros((1, x.shape[1], x.shape[2]))
    one_dims = int((x.size(2)/orig_sz)*targ_sz)
    cnt_mask[:, one_dims:one_dims*2, one_dims:one_dims*2].add_(1)
    return cnt_mask

def _add_center_attn_mask(x):
    "x: tensor, adds a center mask channel"
    cnt_ones = _create_center_mask(x)
    return torch.cat([x, cnt_ones], dim=0)


center_crop = TfmPixel(_rand_center_crop); center_crop.order = 0
center_crop_and_resize = TfmPixel(_rand_center_crop_and_resize); center_crop_and_resize.order = 0
center_attn_mask = TfmPixel(_add_center_attn_mask); center_attn_mask.order = 9999