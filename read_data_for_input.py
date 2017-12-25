import os
from scipy.io import wavfile
import re

file_path = lambda: os.path.dirname(__file__)
train_audio_path = lambda: file_path() + "/data/train/audio/"
test_audio_path = lambda: file_path() + "/data/test/audio/"
train_data_list_file_path = lambda: file_path() + "/data/train/repo_list.txt"

# read data
def read_file_to_sample(one_audio_file_path):
    sample_rate, sample = wavfile.read(one_audio_file_path, 'rb')
    return sample


class ReadTrainData(object):
    def __init__(self, train_file_path):
        self.train_file_path = train_file_path

    def get_data_With_labels(self):
        dataWithLabels = []
        dirTree = os.walk(self.train_file_path)
        labels = []
        index = 0
        for treeNode in dirTree:
            if index == 0:
                labels = treeNode[1]
            else:
                fileParentPath = treeNode[0]
                filePaths = filter(lambda x: x != '',
                                   map(lambda file: fileParentPath + "/" + file if re.match(r".*wav\Z", file) else '',
                                       treeNode[2]))
                currentLabel = labels[index - 1]
                filePathsWithCurrentLabel = [(currentLabel, filePath) for filePath in filePaths]
                dataWithLabels.extend(filePathsWithCurrentLabel)
            index = index + 1

        return dataWithLabels

    def write_list_to_train_folder(self, list_file_path):
        file_list_repo = open(list_file_path, 'w')
        dataWithLables = self.get_data_With_labels()
        for data in dataWithLables:
            file_list_repo.write(str(data))
            file_list_repo.write("\n")
        file_list_repo.close()


if __name__ == '__main__':
    print train_audio_path()
    reader = ReadTrainData(train_audio_path())
    reader.write_list_to_train_folder(train_data_list_file_path())
