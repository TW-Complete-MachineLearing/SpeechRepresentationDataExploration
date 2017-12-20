import os
from scipy import signal
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

train_audio_path = "./data/train/audio/"

class ReadTrainData(object):
    def __init__(self, train_file_path):
        self.train_file_path = train_file_path

    def getDataWithLabels(self):
        dataWithLabels = []
        dirTree = os.walk(self.train_file_path)
        labels = []
        index = 0
        for treeNode in dirTree:
            if index == 0:
                labels = treeNode[1]
            else:
                fileParentPath = treeNode[0]
                filePaths = map(lambda file : fileParentPath + "/" + file, treeNode[2])
                currentLabel = labels[index - 1]
                filePathsWithCurrentLabel = [(currentLabel, filePath) for filePath in filePaths]
                dataWithLabels.extend(filePathsWithCurrentLabel)
            index = index + 1

        return dataWithLabels

    def write_list_to_train_folder(self, list_file_path):
        file_list_repo = open(list_file_path, 'w')
        dataWithLables = self.getDataWithLabels()
        for data in dataWithLables:
            file_list_repo.write(str(data))
            file_list_repo.write("\n")
        file_list_repo.close()


class ShowAudio(object):
    def __init__(self, label, audio_file_path):
        self.audio_file_path = audio_file_path
        self.label = label

    def log_specgram(self, audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
        nperseg = int(round(window_size * sample_rate / 1e3))
        noverlap = int(round(step_size * sample_rate / 1e3))
        freqs, times, spec = signal.spectrogram(audio,
                                                fs=sample_rate,
                                                window='hann',
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                detrend=False)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)

    def show(self):
        sample_rate, samples = wavfile.read(self.audio_file_path)
        freqs, times, spectrogram = self.log_specgram(samples, sample_rate)
        print freqs, times, spectrogram

        fig = plt.figure(figsize=(14, 8))
        ax1 = fig.add_subplot(211)
        ax1.set_title('Raw wave of ' + self.label)
        ax1.set_ylabel('Amplitude')
        ax1.plot(np.linspace(0, sample_rate / len(samples), sample_rate), samples)

        ax2 = fig.add_subplot(212)
        ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
                   extent=[times.min(), times.max(), freqs.min(), freqs.max()])
        ax2.set_yticks(freqs[::16])
        ax2.set_xticks(times[::16])
        ax2.set_title('Spectrogram of ' + self.label)
        ax2.set_ylabel('Freqs in Hz')
        ax2.set_xlabel('Seconds')
        plt.legend()
        plt.show()




if __name__=='__main__':
    reader = ReadTrainData(train_audio_path)
    reader.write_list_to_train_folder('./data/train/repo_list.txt')
    list = reader.getDataWithLabels()
    showers = [ShowAudio(label, fileName) for (label, fileName) in list]
    for shower in showers:
        shower.show()




