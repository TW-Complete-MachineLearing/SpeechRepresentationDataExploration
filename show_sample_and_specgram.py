from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import read_data_for_input as read_data


def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                            fs=sample_rate,
                                            window='hann',
                                            nperseg=nperseg,
                                            noverlap=noverlap,
                                            detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


class ShowSampleAudio(object):
    def __init__(self, sample, label):
        self.sample = sample
        self.label = label
        self.fig = plt.figure(figsize=(14, 8))

    def draw_sample(self):
        ax1 = self.fig.add_subplot(211)
        ax1.set_title('Raw wave of ' + self.label)
        ax1.set_ylabel('Amplitude')
        ax1.plot(np.linspace(0, 1, len(self.sample)), self.sample)

    def draw_spectrogram(self, freqs, times, spectrogram):
        ax2 = self.fig.add_subplot(212)
        ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
                   extent=[times.min(), times.max(), freqs.min(), freqs.max()])
        ax2.set_yticks(freqs[::16])
        ax2.set_xticks(times[::16])
        ax2.set_title('Spectrogram of ' + self.label)
        ax2.set_ylabel('Freqs in Hz')
        ax2.set_xlabel('Seconds')

    def show(self):
        self.draw_sample()
        freqs, times, spectrogram = log_specgram(self.sample, len(self.sample))
        self.draw_spectrogram(freqs, times, spectrogram)
        self.fig.show()


if __name__ == '__main__':
    reader = read_data.ReadTrainData(read_data.train_audio_path())
    data_with_labels = reader.getDataWithLabels()
    index = 1
    show_case_limit = 10
    for (label, file_path) in data_with_labels:
        if index > show_case_limit:
            break
        else:
            index = index + 1
            print label, file_path, "\n"
            sample = read_data.read_file_to_sample(file_path)
            print sample
            shower = ShowSampleAudio(sample, label)
            shower.show()
    raw_input()



