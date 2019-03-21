from fastai.vision import *
from fastai.vision.models import cadene_models 
import exifread

def get_img_metadata(fn):
    "return image metadata"
    f = open(fn, 'rb')
    metadata = {}
    tags = exifread.process_file(f)
    for tag in tags.keys():
        if tag nota in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
            metadata[tag] = tags[tag]
    return metadata

def get_arch_by_name(arch_name):
    "grab model function by name"
    if hasattr(models, arch_name):
        return getattr(models, arch_name)
    elif hasattr(cadene_models, arch_name):
        return getattr(cadene_models, arch_name)
    else:
        raise Exception("Model not found in 'models' of 'cadene_models'")