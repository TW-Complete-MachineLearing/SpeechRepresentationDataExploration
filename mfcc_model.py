import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import read_data_for_input as read_data
import vad_remove_slience as vad


class mfcc_model(object):
    def __init__(self, sample):
        self.sample = sample

    def get_mel_power_spectrogram(self):
        S = librosa.feature.melspectrogram(self.sample, sr=len(self.sample), n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        return log_S

    def show_mel_power_spectrogram(self):
        log_S = self.get_mel_power_spectrogram()
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(log_S, sr=len(self.sample), x_axis='time', y_axis='mel')
        plt.title('Mel power spectrogram ')
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
        plt.show()

    def get_first_two_powerest_spectrograms(self):
        log_S = self.get_mel_power_spectrogram()
        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        return delta2_mfcc

    def show_first_two_powerest_spectrograms(self):
        delta2_mfcc = self.get_first_two_powerest_spectrograms()
        print delta2_mfcc, type(delta2_mfcc), delta2_mfcc.shape
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(delta2_mfcc)
        plt.ylabel('MFCC coeffs')
        plt.xlabel('Time')
        plt.title('MFCC')
        plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    reader = read_data.ReadTrainData(read_data.train_audio_path())
    data_with_labels = reader.get_data_With_labels()
    index = 1
    show_case_limit = 20
    for (label, file_path) in data_with_labels:
        if index > show_case_limit:
            break
        else:
            index = index + 1
            print label, file_path, "\n"
            sample = read_data.read_file_to_sample(file_path)
            print sample
            vader = vad.vad_remove_silence(sample, 16000)
            vad_sample = vader.get_vad_sample()
            mfccer = mfcc_model(vad_sample)
            mfccer.show_first_two_powerest_spectrograms()
    raw_input()
