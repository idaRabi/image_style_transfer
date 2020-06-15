from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from torchvision.datasets.folder import default_loader


class ImageLoader(Dataset):

    def __init__(self, base_path, labels_path, label_to_id, transform, loader=default_loader):
        self.base_path = base_path
        self.labels_path = labels_path
        self.labels_to_id = label_to_id
        self.loader = loader
        self.transform = transform
        self.paths = [join(self.base_path, f) for f in listdir(self.base_path) if isfile(join(self.base_path, f))]
        self.labels = self.read_labels()

    def __getitem__(self, index):
        file_path = self.paths[index]
        file_feature = self.loader(file_path)
        file_feature = self.transform(file_feature)
        file_label = self.labels_to_id[self.labels[file_path]]
        return file_feature, file_label

    def __len__(self):
        return len(self.paths)

    def read_labels(self):
        file = open(self.labels_path)
        lines = file.readlines()
        sep_lines = [x.split("\t") for x in lines]
        result = {}
        for i in range(len(sep_lines)):
            result[join(self.base_path, sep_lines[i][0])] = sep_lines[i][1]
        return result
