from fastai.vision import *
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