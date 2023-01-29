import os
import logging
from torch.utils import data
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# Fields
class Field(object):
    ''' Data fields class.
    '''

    def load(self, data_path, idx, category):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, files):
        ''' Checks if set is complete.

        Args:
            files: files
        '''
        raise NotImplementedError


class Shapes3dDataset(data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=True, transform=None,
                 shared_dict={}, n_views=24, cache_fields=False,
                 split_model_for_images=False):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
            shared_dict (dict): shared dictionary (used for field caching)
            n_views (int): number of views (only relevant when using field
                caching)
            cache_fields(bool): whether to cache fields; this option can be
                useful for small overfitting experiments
            split_model_for_images (bool): whether to split a model by its
                views (can be relevant for small overfitting experiments to
                       perform validation on all views)
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform
        self.cache_fields = cache_fields
        self.n_views = n_views
        self.cached_fields = shared_dict
        self.split_model_for_images = split_model_for_images

        if split_model_for_images:
            assert(n_views > 0)
            print('You are splitting the models by images. Make sure that you entered the correct number of views.')
        #categories=['02691156','02958343','03001627','03211117','04401088','03691459','04379243']
        print (categories)
        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]
        categories.sort()
        categories=['04530566']
        print (categories, 'cccccccccccc')        

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            }

        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, str(split) + '.lst')

            if not os.path.exists(split_file):
                models_c = [f for f in os.listdir(
                    subpath) if os.path.isdir(os.path.join(subpath, f))]
            else:
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
            models_c = list(filter(lambda x: len(x) > 0, models_c))
            if split_model_for_images:
                for m in models_c:
                    for i in range(n_views):
                        self.models += [
                            {'category': c, 'model': m,
                                'category_id': c_idx, 'image_id': i}
                        ]
            else:
                self.models += [
                    {'category': c, 'model': m, 'category_id': c_idx}
                    for m in models_c
                ]

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}
        for field_name, field in self.fields.items():
                    
            #try:
            if 1:            
                if self.cache_fields:
                    if self.split_model_for_images:
                        idx_img = self.models[idx]['image_id']
                    else:
                        idx_img = np.random.randint(0, self.n_views)
                    k = '%s_%s_%d' % (model_path, field_name, idx_img)

                    if k in self.cached_fields:
                        field_data = self.cached_fields[k]
                    else:
                        field_data = field.load(model_path, idx, c_idx,
                                                input_idx_img=idx_img)
                        self.cached_fields[k] = field_data
                else:
                    if self.split_model_for_images:
                        idx_img = self.models[idx]['image_id']
                        field_data = field.load(
                            model_path, idx, c_idx, idx_img)
                    else:
                        #print (model_path, idx, c_idx)                    
                        field_data = field.load(model_path, idx, c_idx)
            '''except Exception:
                if self.no_except:
                    logger.warn(
                        'Error occurred when loading field %s of model %s (%s)'
                        % (field_name, model, category)
                    )
                    return None
                else:
                    raise'''

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.

        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True


def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''

    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
