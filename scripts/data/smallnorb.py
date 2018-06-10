import struct
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from tqdm import tqdm
from os import makedirs
from os.path import join
from os.path import exists
from itertools import groupby


class SmallNORBExample:

    def __init__(self):
        self.image_lt  = None
        self.image_rt  = None
        self.category  = None
        self.instance  = None
        self.elevation = None
        self.azimuth   = None
        self.lighting  = None

    def __lt__(self, other):
        return self.category < other.category or \
                (self.category == other.category and self.instance < other.instance)

    def show(self, subplots):
        fig, axes = subplots
        fig.suptitle(
            'Category: {:02d} - Instance: {:02d} - Elevation: {:02d} - Azimuth: {:02d} - Lighting: {:02d}'.format(
                self.category, self.instance, self.elevation, self.azimuth, self.lighting))
        axes[0].imshow(self.image_lt, cmap='gray')
        axes[1].imshow(self.image_rt, cmap='gray')

    @property
    def pose(self):
        return np.array([self.elevation, self.azimuth, self.lighting], dtype=np.float32)


class SmallNORBDataset:

    # Number of examples in both train and test set
    n_examples = 24300

    # Categories present in small NORB dataset
    categories = ['animal', 'human', 'airplane', 'truck', 'car']

    def __init__(self, dataset_root):
        """
        Initialize small NORB dataset wrapper
        
        Parameters
        ----------
        dataset_root: str
            Path to directory where small NORB archives have been extracted.
        """

        self.dataset_root = dataset_root
        self.initialized  = False

        # Store path for each file in small NORB dataset (for compatibility the original filename is kept)
        self.dataset_files = {
            'train': {
                'cat':  join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat'),
                'info': join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat'),
                'dat':  join(self.dataset_root, 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')
            },
            'test':  {
                'cat':  join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat'),
                'info': join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat'),
                'dat':  join(self.dataset_root, 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat')
            }
        }

        # Initialize both train and test data structures
        self.data = {
            'train': [SmallNORBExample() for _ in range(SmallNORBDataset.n_examples)],
            'test':  [SmallNORBExample() for _ in range(SmallNORBDataset.n_examples)]
        }

        # Fill data structures parsing dataset binary files
        for data_split in ['train', 'test']:
            self._fill_data_structures(data_split)

        self.initialized = True

    def explore_random_examples(self, dataset_split):
        """
        Visualize random examples for dataset exploration purposes
        
        Parameters
        ----------
        dataset_split: str
            Dataset split, can be either 'train' or 'test'

        Returns
        -------
        None
        """
        if self.initialized:
            subplots = plt.subplots(nrows=1, ncols=2)
            for i in np.random.permutation(SmallNORBDataset.n_examples):
                self.data[dataset_split][i].show(subplots)
                plt.waitforbuttonpress()
                plt.cla()

    def export_to_jpg(self, export_dir):
        """
        Export all dataset images to `export_dir` directory
        
        Parameters
        ----------
        export_dir: str
            Path to export directory (which is created if nonexistent)
            
        Returns
        -------
        None
        """
        if self.initialized:
            print(('Exporting images to {}...'.format(export_dir))) #end='', flush=True)
            for split_name in ['train', 'test']:

                split_dir = join(export_dir, split_name)
                if not exists(split_dir):
                    makedirs(split_dir)

                for i, norb_example in enumerate(self.data[split_name]):

                    category = SmallNORBDataset.categories[norb_example.category]
                    instance = norb_example.instance

                    image_lt_path = join(split_dir, '{:06d}_{}_{:02d}_lt.jpg'.format(i, category, instance))
                    image_rt_path = join(split_dir, '{:06d}_{}_{:02d}_rt.jpg'.format(i, category, instance))

                    scipy.misc.imsave(image_lt_path, norb_example.image_lt)
                    scipy.misc.imsave(image_rt_path, norb_example.image_rt)
            print('Done.')
    
    def group_dataset_by_category_and_instance(self, dataset_split):
        """
        Group small NORB dataset for (category, instance) key
        
        Parameters
        ----------
        dataset_split: str
            Dataset split, can be either 'train' or 'test'

        Returns
        -------
        groups: list
            List of 25 groups of 972 elements each. All examples of each group are
            from the same category and instance
        """
        if dataset_split not in ['train', 'test']:
            raise ValueError('Dataset split "{}" not allowed.'.format(dataset_split))

        groups = []
        for key, group in groupby(iterable=sorted(self.data[dataset_split]),
                                  key=lambda x: (x.category, x.instance)):
            groups.append(list(group))

        return groups

    def _fill_data_structures(self, dataset_split):
        """
        Fill SmallNORBDataset data structures for a certain `dataset_split`.
        
        This means all images, category and additional information are loaded from binary
        files of the current split.
        
        Parameters
        ----------
        dataset_split: str
            Dataset split, can be either 'train' or 'test'

        Returns
        -------
        None

        """
        dat_data  = self._parse_NORB_dat_file(self.dataset_files[dataset_split]['dat'])
        cat_data  = self._parse_NORB_cat_file(self.dataset_files[dataset_split]['cat'])
        info_data = self._parse_NORB_info_file(self.dataset_files[dataset_split]['info'])
        for i, small_norb_example in enumerate(self.data[dataset_split]):
            small_norb_example.image_lt   = dat_data[2 * i]
            small_norb_example.image_rt   = dat_data[2 * i + 1]
            small_norb_example.category  = cat_data[i]
            small_norb_example.instance  = info_data[i][0]
            small_norb_example.elevation = info_data[i][1]
            small_norb_example.azimuth   = info_data[i][2]
            small_norb_example.lighting  = info_data[i][3]

    @staticmethod
    def matrix_type_from_magic(magic_number):
        """
        Get matrix data type from magic number
        See here: https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/readme for details.

        Parameters
        ----------
        magic_number: tuple
            First 4 bytes read from small NORB files 

        Returns
        -------
        element type of the matrix
        """
        convention = {'1E3D4C51': 'single precision matrix',
                      '1E3D4C52': 'packed matrix',
                      '1E3D4C53': 'double precision matrix',
                      '1E3D4C54': 'integer matrix',
                      '1E3D4C55': 'byte matrix',
                      '1E3D4C56': 'short matrix'}
        magic_str = bytearray(reversed(magic_number)).hex().upper()
        return convention[magic_str]

    @staticmethod
    def _parse_small_NORB_header(file_pointer):
        """
        Parse header of small NORB binary file
        
        Parameters
        ----------
        file_pointer: BufferedReader
            File pointer just opened in a small NORB binary file

        Returns
        -------
        file_header_data: dict
            Dictionary containing header information
        """
        # Read magic number
        magic = struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

        file_header_data = {'magic_number': magic,
                            'matrix_type': SmallNORBDataset.matrix_type_from_magic(magic),
                            'dimensions': dimensions}
        return file_header_data

    @staticmethod
    def _parse_NORB_cat_file(file_path):
        """
        Parse small NORB category file
        
        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-cat.mat` file

        Returns
        -------
        examples: ndarray
            Ndarray of shape (24300,) containing the category of each example
        """
        with open(file_path, mode='rb') as f:
            header = SmallNORBDataset._parse_small_NORB_header(f)

            num_examples, = header['dimensions']

            struct.unpack('<BBBB', f.read(4))  # ignore this integer
            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            examples = np.zeros(shape=num_examples, dtype=np.int32)
            for i in tqdm(list(range(num_examples)), desc='Loading categories...'):
                category, = struct.unpack('<i', f.read(4))
                examples[i] = category

            return examples

    @staticmethod
    def _parse_NORB_dat_file(file_path):
        """
        Parse small NORB data file

        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-dat.mat` file

        Returns
        -------
        examples: ndarray
            Ndarray of shape (48600, 96, 96) containing images couples. Each image couple
            is stored in position [i, :, :] and [i+1, :, :]
        """
        with open(file_path, mode='rb') as f:

            header = SmallNORBDataset._parse_small_NORB_header(f)

            num_examples, channels, height, width = header['dimensions']

            examples = np.zeros(shape=(num_examples * channels, height, width), dtype=np.uint8)

            for i in tqdm(list(range(num_examples * channels)), desc='Loading images...'):

                # Read raw image data and restore shape as appropriate
                image = struct.unpack('<' + height * width * 'B', f.read(height * width))
                image = np.uint8(np.reshape(image, newshape=(height, width)))

                examples[i] = image

        return examples

    @staticmethod
    def _parse_NORB_info_file(file_path):
        """
        Parse small NORB information file

        Parameters
        ----------
        file_path: str
            Path of the small NORB `*-info.mat` file

        Returns
        -------
        examples: ndarray
            Ndarray of shape (24300,4) containing the additional info of each example.
            
             - column 1: the instance in the category (0 to 9)
             - column 2: the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70 
               degrees from the horizontal respectively)
             - column 3: the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
             - column 4: the lighting condition (0 to 5)
        """
        with open(file_path, mode='rb') as f:

            header = SmallNORBDataset._parse_small_NORB_header(f)

            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            num_examples, num_info = header['dimensions']

            examples = np.zeros(shape=(num_examples, num_info), dtype=np.int32)

            for r in tqdm(list(range(num_examples)), desc='Loading info...'):
                for c in range(num_info):
                    info, = struct.unpack('<i', f.read(4))
                    examples[r, c] = info

        return examples
