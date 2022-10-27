import torch
import os

from torch.utils import data
from torchvision import datasets
from torchvision import transforms

class ColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./Data', split='train1', transform=None, target_transform=None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)
    

    env = split
    self.prepare_colored_mnist()
    if env in ['train1', 'train2', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
    elif env == 'all_train':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img,env,target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img,env,target

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
      
    import numpy as np
    from PIL import Image
    
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      if np.random.uniform() < 0.25:
        binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < 20000:
        # 20% in the first training environment
        if np.random.uniform() < 0.4:
          color_red = not color_red
      elif idx < 40000:
        # 10% in the first training environment
        if np.random.uniform() < 0.1:
          color_red = not color_red
      else:
        # 90% in the test environment
        if np.random.uniform() < 0.9:
          color_red = not color_red


      def color_grayscale_arr(arr, red=True):
        """Converts grayscale image to either red or green"""
        assert arr.ndim == 2
        dtype = arr.dtype
        h, w = arr.shape
        arr = np.reshape(arr, [h, w, 1])
        if red:
          arr = np.concatenate([arr,
                                np.zeros((h, w, 2), dtype=dtype)], axis=2)
        else:
          arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                                arr,
                                np.zeros((h, w, 1), dtype=dtype)], axis=2)
        return arr

      colored_arr = color_grayscale_arr(im_array, red=color_red)

      if idx < 20000:
        train1_set.append((Image.fromarray(colored_arr),0, binary_label))
      elif idx < 40000:
        train2_set.append((Image.fromarray(colored_arr),1, binary_label))
      else:
        test_set.append((Image.fromarray(colored_arr),2, binary_label))


    os.mkdir(colored_mnist_dir)
    torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
    torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
    torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))

class Four_ColoredMNIST(datasets.VisionDataset):
  def __init__(self,args, root='./Data', split='train1', transform=None, target_transform=None):
    super().__init__(root, transform=transform,
                                target_transform=target_transform)
    

    env = split
    self.args = args
    self.prepare_colored_mnist()
    if env in ['train1', 'train2', 'train3', 'train4', 'test', 'gray_train1', 'gray_train2', 'gray_train3', 'gray_train4', 'gray_test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
    elif env == 'all_train':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'train3.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'train4.pt'))
    elif env == 'gray_all_train':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'gray_train1.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'gray_train2.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'gray_train3.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'gray_test.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST', 'gray_train4.pt'))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, train3, train4, test, and gray_train1, gray_train2, gray_train3, gray_train4, gray_test, all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img,env,target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img,env,target

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
      
    import numpy as np
    from PIL import Image
    
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train3.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train4.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'gray_train1.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'gray_train2.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'gray_train3.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'gray_train4.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'gray_test.pt')) :
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Colored MNIST')
    if self.args.type_datasets == 'MNIST':
      train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)
    elif self.args.type_datasets == 'FashionMNIST':
      train_mnist = datasets.mnist.FashionMNIST(self.root, train=True, download=True)
    elif self.args.type_datasets == 'KMNIST':
      train_mnist = datasets.mnist.KMNIST(self.root, train=True, download=True)
    elif self.args.type_datasets == 'EMNIST':
      train_mnist = datasets.mnist.EMNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    train3_set = []
    train4_set = []
    test_set = []
    gray_train1_set = []
    gray_train2_set = []
    gray_train3_set = []
    gray_train4_set = []
    gray_test_set = []

    for idx, (im, label) in enumerate(train_mnist):
      if idx % 10000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      if np.random.uniform() < 0.25:
        binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < 10000:
        # 40% in the first training environment
        if np.random.uniform() < 0.4:
          color_red = not color_red
      elif idx < 20000:
        # 10% in the first training environment
        if np.random.uniform() < 0.3:
          color_red = not color_red
      elif idx < 30000:
        # 10% in the first training environment
        if np.random.uniform() < 0.2:
          color_red = not color_red
      elif idx < 40000:
        # 10% in the first training environment
        if np.random.uniform() < 0.1:
          color_red = not color_red
      else:
        # 90% in the test environment
        if np.random.uniform() < 0.9:
          color_red = not color_red

      def color_grayscale_arr(arr, red=True):
        """Converts grayscale image to either red or green"""
        assert arr.ndim == 2
        dtype = arr.dtype
        h, w = arr.shape
        arr = np.reshape(arr, [h, w, 1])
        if red:
          arr = np.concatenate([arr,
                                np.zeros((h, w, 2), dtype=dtype)], axis=2)
        else:
          arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                                arr,
                                np.zeros((h, w, 1), dtype=dtype)], axis=2)
        return arr
      
      def get_grayscale_image(arr):
        """get_grayscale_image"""
        assert arr.ndim == 2
        dtype = arr.dtype
        h,w = arr.shape
        arr = np.reshape(arr,[h, w, 1])
        arr = np.concatenate([arr, arr,np.zeros((h, w, 1), dtype=dtype) ], axis=2)
        return arr

      gray_arr = get_grayscale_image(im_array)

      colored_arr = color_grayscale_arr(im_array, red=color_red)
      #More environment
      if idx < 10000:
        train1_set.append((Image.fromarray(colored_arr),0, binary_label))
        gray_train1_set.append((Image.fromarray(gray_arr),0, binary_label))
      elif idx < 20000:
        train2_set.append((Image.fromarray(colored_arr),1, binary_label))
        gray_train2_set.append((Image.fromarray(gray_arr),1, binary_label))
      elif idx < 30000:
        train3_set.append((Image.fromarray(colored_arr),2, binary_label))
        gray_train3_set.append((Image.fromarray(gray_arr),2, binary_label))
      elif idx < 40000:
        train4_set.append((Image.fromarray(colored_arr),3, binary_label))
        gray_train4_set.append((Image.fromarray(gray_arr),3, binary_label))
      else:
        test_set.append((Image.fromarray(colored_arr),4, binary_label))
        gray_test_set.append((Image.fromarray(gray_arr),4, binary_label))


    os.mkdir(colored_mnist_dir)
    torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
    torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
    torch.save(train3_set, os.path.join(colored_mnist_dir, 'train3.pt'))
    torch.save(train4_set, os.path.join(colored_mnist_dir, 'train4.pt'))
    torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))
    torch.save(gray_train1_set, os.path.join(colored_mnist_dir, 'gray_train1.pt'))
    torch.save(gray_train2_set, os.path.join(colored_mnist_dir, 'gray_train2.pt'))
    torch.save(gray_train3_set, os.path.join(colored_mnist_dir, 'gray_train3.pt'))
    torch.save(gray_train4_set, os.path.join(colored_mnist_dir, 'gray_train4.pt'))
    torch.save(gray_test_set, os.path.join(colored_mnist_dir, 'gray_test.pt'))

class Eight_ColoredMNIST(datasets.VisionDataset):
  def __init__(self, root='./Data', split='train1', transform=None, target_transform=None):
    super().__init__(root, transform=transform,
                                target_transform=target_transform)
    

    env = split
    self.prepare_colored_mnist()
    if env in ['train1', 'train2', 'train3', 'train4','train5', 'train6', 'train7', 'train8', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST_8', env) + '.pt')
    elif env == 'all_train':
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST_8', 'train1.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST_8', 'train2.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST_8', 'train3.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST_8', 'train4.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST_8', 'train5.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST_8', 'train6.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST_8', 'train7.pt')) + \
                               torch.load(os.path.join(self.root, 'ColoredMNIST_8', 'train8.pt'))
    else:
      raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, train3, train4, train5, train6, train7, train8, test, and all_train')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img,env,target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img,env,target

  def __len__(self):
    return len(self.data_label_tuples)

  def prepare_colored_mnist(self):
      
    import numpy as np
    from PIL import Image
    
    colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST_8')
    if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train3.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train4.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train5.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train6.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train7.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'train8.pt')) \
        and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
      print('Colored MNIST dataset already exists')
      return

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)
    # train_mnist = datasets.mnist.FashionMNIST(self.root, train=True, download=True)
    # train_mnist = datasets.mnist.KMNIST(self.root, train=True, download=True)
    # train_mnist = datasets.mnist.EMNIST(self.root, train=True, download=True)

    train1_set = []
    train2_set = []
    train3_set = []
    train4_set = []
    train5_set = []
    train6_set = []
    train7_set = []
    train8_set = []
    test_set = []
    for idx, (im, label) in enumerate(train_mnist):
      if idx % 5000 == 0:
        print(f'Converting image {idx}/{len(train_mnist)}')
      im_array = np.array(im)

      # Assign a binary label y to the image based on the digit
      binary_label = 0 if label < 5 else 1

      # Flip label with 25% probability
      if np.random.uniform() < 0.25:
        binary_label = binary_label ^ 1

      # Color the image either red or green according to its possibly flipped label
      color_red = binary_label == 0

      # Flip the color with a probability e that depends on the environment
      if idx < 5000:
        # 40% in the first training environment
        if np.random.uniform() < 0.4:
          color_red = not color_red
      elif idx < 10000:
        # 35% in the first training environment
        if np.random.uniform() < 0.35:
          color_red = not color_red
      elif idx < 15000:
        # 30% in the first training environment
        if np.random.uniform() < 0.3:
          color_red = not color_red
      elif idx < 20000:
        # 25% in the first training environment
        if np.random.uniform() < 0.25:
          color_red = not color_red
      elif idx < 25000:
        # 20% in the first training environment
        if np.random.uniform() < 0.2:
          color_red = not color_red
      elif idx < 30000:
        # 15% in the first training environment
        if np.random.uniform() < 0.15:
          color_red = not color_red
      elif idx < 35000:
        # 10% in the first training environment
        if np.random.uniform() < 0.1:
          color_red = not color_red
      elif idx < 40000:
        # 5% in the first training environment
        if np.random.uniform() < 0.05:
          color_red = not color_red
      else:
        # 90% in the test environment
        if np.random.uniform() < 0.9:
          color_red = not color_red

      def color_grayscale_arr(arr, red=True):
        """Converts grayscale image to either red or green"""
        assert arr.ndim == 2
        dtype = arr.dtype
        h, w = arr.shape
        arr = np.reshape(arr, [h, w, 1])
        if red:
          arr = np.concatenate([arr,
                                np.zeros((h, w, 2), dtype=dtype)], axis=2)
        else:
          arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),
                                arr,
                                np.zeros((h, w, 1), dtype=dtype)], axis=2)
        return arr

      colored_arr = color_grayscale_arr(im_array, red=color_red)
      #More environment
      if idx < 5000:
        train1_set.append((Image.fromarray(colored_arr),0, binary_label))
      elif idx < 10000:
        train2_set.append((Image.fromarray(colored_arr),1, binary_label))
      elif idx < 15000:
        train3_set.append((Image.fromarray(colored_arr),2, binary_label))
      elif idx < 20000:
        train4_set.append((Image.fromarray(colored_arr),3, binary_label))
      elif idx < 25000:
        train5_set.append((Image.fromarray(colored_arr),4, binary_label))
      elif idx < 30000:
        train6_set.append((Image.fromarray(colored_arr),5, binary_label))
      elif idx < 35000:
        train7_set.append((Image.fromarray(colored_arr),6, binary_label))
      elif idx < 40000:
        train8_set.append((Image.fromarray(colored_arr),7, binary_label))
      else:
        test_set.append((Image.fromarray(colored_arr),8, binary_label))


    os.mkdir(colored_mnist_dir)
    torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
    torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
    torch.save(train3_set, os.path.join(colored_mnist_dir, 'train3.pt'))
    torch.save(train4_set, os.path.join(colored_mnist_dir, 'train4.pt'))
    torch.save(train5_set, os.path.join(colored_mnist_dir, 'train5.pt'))
    torch.save(train6_set, os.path.join(colored_mnist_dir, 'train6.pt'))
    torch.save(train7_set, os.path.join(colored_mnist_dir, 'train7.pt'))
    torch.save(train8_set, os.path.join(colored_mnist_dir, 'train8.pt'))
    torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))

def load_data_mnist(batch_size):
    trans = transforms.ToTensor()
    mnist_train_1 = ColoredMNIST(root='../data',split='train1',transform=trans)
    mnist_train_2 = ColoredMNIST(root='../data',split='train2',transform=trans)
    mnist_train_all = ColoredMNIST(root='../data',split='all_train',transform=trans)
    mnist_test = ColoredMNIST(root='../data',split='test',transform=trans)
    train_iter_1 = data.DataLoader(mnist_train_1,batch_size,shuffle=True)
    train_iter_2 = data.DataLoader(mnist_train_2,batch_size,shuffle=True)
    train_iter_all = data.DataLoader(mnist_train_all,batch_size,shuffle=True)
    test_iter = data.DataLoader(mnist_test,batch_size,shuffle=False)
    return (train_iter_1,train_iter_2,train_iter_all),test_iter

def load_data_more_mnist_gray(args,batch_size):
    trans = transforms.ToTensor()
    mnist_train_all = Four_ColoredMNIST(args,root='../data',split='gray_all_train',transform=trans)
    train_iter_all = data.DataLoader(mnist_train_all,batch_size,shuffle=True)
    return train_iter_all

def load_data_more_mnist(args):
    batch_size = args.batch_size
    trans = transforms.ToTensor()
    mnist_train_1 = Four_ColoredMNIST(args,root='../data',split='train1',transform=trans)
    mnist_train_2 = Four_ColoredMNIST(args,root='../data',split='train2',transform=trans)
    mnist_train_3 = Four_ColoredMNIST(args,root='../data',split='train3',transform=trans)
    mnist_train_4 = Four_ColoredMNIST(args,root='../data',split='train4',transform=trans)
    mnist_train_all = Four_ColoredMNIST(args,root='../data',split='all_train',transform=trans)
    mnist_test = Four_ColoredMNIST(args,root='../data',split='test',transform=trans)
    train_iter_1 = data.DataLoader(mnist_train_1,batch_size,shuffle=True)
    train_iter_2 = data.DataLoader(mnist_train_2,batch_size,shuffle=True)
    train_iter_3 = data.DataLoader(mnist_train_3,batch_size,shuffle=True)
    train_iter_4 = data.DataLoader(mnist_train_4,batch_size,shuffle=True)
    train_iter_all = data.DataLoader(mnist_train_all,batch_size,shuffle=True)
    test_iter = data.DataLoader(mnist_test,batch_size,shuffle=False)
    return (train_iter_1,train_iter_2,train_iter_3,train_iter_4,train_iter_all),test_iter

def load_data_more_mnist_2(batch_size):
    trans = transforms.ToTensor()
    mnist_train_1 = Eight_ColoredMNIST(root='../data',split='train1',transform=trans)
    mnist_train_2 = Eight_ColoredMNIST(root='../data',split='train2',transform=trans)
    mnist_train_3 = Eight_ColoredMNIST(root='../data',split='train3',transform=trans)
    mnist_train_4 = Eight_ColoredMNIST(root='../data',split='train4',transform=trans)
    mnist_train_5 = Eight_ColoredMNIST(root='../data',split='train5',transform=trans)
    mnist_train_6 = Eight_ColoredMNIST(root='../data',split='train6',transform=trans)
    mnist_train_7 = Eight_ColoredMNIST(root='../data',split='train7',transform=trans)
    mnist_train_8 = Eight_ColoredMNIST(root='../data',split='train8',transform=trans)
    mnist_train_all = Eight_ColoredMNIST(root='../data',split='all_train',transform=trans)
    mnist_test = Eight_ColoredMNIST(root='../data',split='test',transform=trans)
    train_iter_1 = data.DataLoader(mnist_train_1,batch_size,shuffle=True)
    train_iter_2 = data.DataLoader(mnist_train_2,batch_size,shuffle=True)
    train_iter_3 = data.DataLoader(mnist_train_3,batch_size,shuffle=True)
    train_iter_4 = data.DataLoader(mnist_train_4,batch_size,shuffle=True)
    train_iter_5 = data.DataLoader(mnist_train_5,batch_size,shuffle=True)
    train_iter_6 = data.DataLoader(mnist_train_6,batch_size,shuffle=True)
    train_iter_7 = data.DataLoader(mnist_train_7,batch_size,shuffle=True)
    train_iter_8 = data.DataLoader(mnist_train_8,batch_size,shuffle=True)
    train_iter_all = data.DataLoader(mnist_train_all,batch_size,shuffle=True)
    test_iter = data.DataLoader(mnist_test,batch_size,shuffle=False)
    return (train_iter_1,train_iter_2,train_iter_3,train_iter_4,train_iter_5,train_iter_6,train_iter_7,train_iter_8,train_iter_all),test_iter