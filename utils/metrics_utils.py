import os
import os.path as osp
from PIL import Image
import torch
import json
from torchvision import transforms

    
class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

class DummyDataset(torch.utils.data.Dataset):
    FLAGS = ['img', 'txt']
    def __init__(self, real_path, fake_path,
                real_flag: str = 'img',
                fake_flag: str = 'txt',
                transform = None,
                tokenizer = None) -> None:
        super().__init__()
        assert real_flag in self.FLAGS and fake_flag in self.FLAGS, \
            'CLIP Score only support modality of {}. However, get {} and {}'.format(
                self.FLAGS, real_flag, fake_flag
            )
        self.real_folder = self._combine_without_prefix(real_path)
        self.real_flag = real_flag
        self.fake_folder = self._combine_without_prefix(fake_path)
        self.fake_flag = fake_flag
        self.transform = transform
        self.tokenizer = tokenizer
        # assert self._check()

    def __len__(self):
        return len(self.real_folder)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_path = self.real_folder[index]
        fake_path = self.fake_folder[index]
        real_data = self._load_modality(real_path, self.real_flag)
        fake_data = self._load_modality(fake_path, self.fake_flag)

        sample = dict(real=real_data, fake=fake_data)
        return sample
    
    def _load_modality(self, path, modality):
        if modality == 'img':
            data = self._load_img(path)
        elif modality == 'txt':
            data = self._load_txt(path)
        else:
            raise TypeError("Got unexpected modality: {}".format(modality))
        return data

    def _load_img(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def _load_txt(self, path):
        with open(path, 'r') as fp:
            data = fp.read()
            fp.close()
        if self.tokenizer is not None:
            data = self.tokenizer(data).squeeze()
        return data

    def _check(self):
        for idx in range(len(self)):
            real_name = self.real_folder[idx].split('.')
            fake_name = self.fake_folder[idx].split('.')
            if fake_name != real_name:
                return False
        return True
    
    def _combine_without_prefix(self, folder_path, prefix='.'):
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(osp.join(folder_path, name))
        folder.sort()
        return folder
    
class JsonDataset(torch.utils.data.Dataset):
    FLAGS = ['img', 'txt']
    def __init__(self, jsonl_path,
                real_flag: str = 'img',
                fake_flag: str = 'txt',
                transform = None,
                tokenizer = None) -> None:
        super().__init__()
        self.real_folder, self.fake_folder = self._read_jsonl(jsonl_path)
        self.real_flag = real_flag
        self.fake_flag = fake_flag

        self.transform = transform
        self.tokenizer = tokenizer
        # self._check()

    def __len__(self):
        return len(self.real_folder)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_path = self.real_folder[index]
        fake_path = self.fake_folder[index]

        real_data = self._load_modality(real_path, self.real_flag)
        fake_data = self._load_modality(fake_path, self.fake_flag)

        sample = dict(real=real_data, fake=fake_data)
        return sample
    
    def _load_modality(self, path, modality):
        if modality == 'img':
            assert path.split('.')[-1] in ['jpg', 'png'], "Got unexpected image format: {}".format(path.split('.')[-1])
            data = self._load_img(path)
        elif modality == 'txt':
            data = self._load_txt(path)
        else:
            raise TypeError("Got unexpected modality: {}".format(modality))
        return data

    def _load_img(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def _load_txt(self, path):
        if path.split('.')[-1] == 'txt':
            with open(path, 'r') as fp:
                data = fp.read()
                fp.close()
        else:
            data = path
        if self.tokenizer is not None:
            data = self.tokenizer(data).squeeze()
        return data

    def _check(self):
        assert len(self.real_folder) == len(self.fake_folder), "The number of real and fake images must be the same."
        # 要改
        for idx in range(len(self)):
            real_name = self.real_folder[idx].split('.')
            fake_name = self.fake_folder[idx].split('.')
            if fake_name != real_name:
                return False
        return True
    
    def _combine_without_prefix(self, folder_path, prefix='.'):
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(osp.join(folder_path, name))
        folder.sort()
        return folder
    
    def _read_jsonl(self, jsonl_path):
        real_folder = []
        fake_folder = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                real_folder.append(data['real_path'].strip())
                fake_folder.append(data['fake_path'].strip())
        return real_folder, fake_folder


if __name__ == '__main__':
    json_path = "/alg_vepfs/private/panfayu/sqj/my_code/T2I-Metrics/examples/img-txt.jsonl"
    dataset = JsonDataset(json_path, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        real = batch['real']
        fake = batch['fake']
        breakpoint()