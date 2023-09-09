import yaml
from im2mesh import data
from im2mesh import dvr
import logging
from multiprocessing import Manager
import os


# method directory; for this project we only use DVR
method_dict = {
    'dvr': dvr,
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, len_dataset=0):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device, len_dataset=len_dataset)
    return model


def set_logger(cfg):
    logfile = os.path.join(cfg['training']['out_dir'],
                           cfg['training']['logfile'])
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s %(name)s: %(message)s',
        datefmt='%m-%d %H:%M',
        filename=logfile,
        filemode='a',
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_handler)


# Trainer
def get_trainer(model, optimizer, cfg, device, generator=None):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    set_logger(cfg)
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device, generator)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator


# Renderer
def get_renderer(model, cfg, device):
    ''' Returns a render instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    renderer = method_dict[method].config.get_renderer(model, cfg, device)
    return renderer


def get_dataset(cfg, mode='train', return_idx=False, return_category=False,
                **kwargs):
    ''' Returns a dataset instance.

    Args:
        cfg (dict): config dictionary
        mode (string): which mode is used (train / val /test / render)
        return_idx (bool): whether to return model index
        return_category (bool): whether to return model category
    '''
    # Get fields with cfg
    method = cfg['method']
    input_type = cfg['data']['input_type']
    dataset_name = cfg['data']['dataset_name']
    dataset_folder = cfg['data']['path']

    categories = cfg['data']['classes']
    cache_fields = cfg['data']['cache_fields']
    n_views = cfg['data']['n_views']
    split_model_for_images = cfg['data']['split_model_for_images']

    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
        'render': cfg['data']['test_split'],
    }
    split = splits[mode]
    fields = method_dict[method].config.get_data_fields(cfg, mode=mode)

    if input_type == 'idx':
        input_field = data.IndexField()
        fields['inputs'] = input_field
    elif input_type == 'image':
        random_view = True if \
            (mode == 'train' or dataset_name == 'NMR') else False
        resize_img_transform = data.ResizeImage(cfg['data']['img_size_input'])
        fields['inputs'] = data.ImagesField(
            cfg['data']['img_folder_input'],
            transform=resize_img_transform,
            with_mask=False, with_camera=False,
            extension=cfg['data']['img_extension_input'],
            n_views=cfg['data']['n_views_input'], random_view=random_view)

    else:
        input_field = None

    if return_idx:
        fields['idx'] = data.IndexField()

    if return_category:
        fields['category'] = data.CategoryField()

    manager = Manager()
    shared_dict = manager.dict()

    if ((dataset_name == 'Shapes3D') or
        (dataset_name == 'DTU') or
            (dataset_name == 'NMR')):
        dataset = data.Shapes3dDataset(
            dataset_folder, fields, split=split,
            categories=categories,
            shared_dict=shared_dict,
            n_views=n_views, cache_fields=cache_fields,
            split_model_for_images=split_model_for_images)
    elif dataset_name == 'images':
        dataset = data.ImageDataset(
            dataset_folder, return_idx=True
        )
    else:
        raise ValueError('Invalid dataset_name!')

    return dataset
