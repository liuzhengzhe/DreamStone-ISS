import os
import urllib
import torch
from torch.utils import model_zoo
import shutil
import datetime


class CheckpointIO(object):
    ''' CheckpointIO class.

    It handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): path where checkpoints are saved
    '''

    def __init__(self, checkpoint_dir='./chkpts', **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        ''' Saves the current module dictionary.

        Args:
            filename (str): name of output file
        '''
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)

        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def backup_model_best(self, filename, **kwargs):
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filename):
            # Backup model
            backup_dir = os.path.join(self.checkpoint_dir, 'backup_model_best')
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            ts = datetime.datetime.now().timestamp()
            filename_backup = os.path.join(backup_dir, '%s.pt' % ts)
            shutil.copy(filename, filename_backup)

    def load(self, filename, device):
        '''Loads a module dictionary from local file or url.

        Args:
            filename (str): name of saved module dictionary
        '''
        filename='model.pt'

        if is_url(filename):
            print ('url',filename)
            return self.load_url(filename, device)
        else:
            import os
            root=os.getcwd()
            print ('local',filename)
            return self.load_file(filename, device)

    def load_file(self, filename, device):
        '''Loads a module dictionary from file.

        Args:
            filename (str): name of saved module dictionary
        '''
        
        #if not os.path.isabs(filename):
        #    filename = os.path.join(self.checkpoint_dir, filename)
        print ('loading name',filename)
        if os.path.exists(filename):
            print('loading name',filename)
            print('=> Loading checkpoint from local file...')
            state_dict = torch.load(filename, map_location=torch.device(device))
            scalars = self.parse_state_dict(state_dict)
            return scalars
        else:
            raise FileExistsError

    def load_url(self, url, device):
        '''Load a module dictionary from url.

        Args:
            url (str): url to saved model
        '''
        print(url)
        print('=> Loading checkpoint from url...')
        state_dict = model_zoo.load_url(url, progress=True, map_location=torch.device(device))
        scalars = self.parse_state_dict(state_dict)
        return scalars

    def parse_state_dict(self, state_dict):
        '''Parse state_dict of model and return scalars.

        Args:
            state_dict (dict): State dict of model
    '''

        for k, v in self.module_dict.items():
            if k in state_dict:
                try:
                  v.load_state_dict(state_dict[k])
                
                except:
                  pass
            else:
                print('Warning: Could not find %s in checkpoint!' % k)
        scalars = {k: v for k, v in state_dict.items()
                   if k not in self.module_dict}
        return scalars


def is_url(url):
    ''' Checks if input string is a URL.

    Args:
        url (string): URL
    '''
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')
