import webrtcvad
import numpy as np
import read_data_for_input as read_data
import matplotlib.pyplot as plt

class vad_remove_silence(object):
    def __init__(self, sample, sample_rate):
        self.sample_rate = sample_rate
        self.sample = sample
        self.vad = webrtcvad.Vad()
        self.init_vad()

    def init_vad(self):
        self.vad.set_mode(3)

    def get_vad_sample(self):
        frames = self.get_frames_by_duration_ms(10)
        vad_sample = np.array([], dtype=np.int16)
        for frame in frames:
            if self.vad.is_speech(frame, self.sample_rate):
                vad_sample = np.append(vad_sample, frame)
            else:
                continue

        return vad_sample


    def get_frames_by_duration_ms(self, frame_duration_ms):
        n = int(self.sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / self.sample_rate) / 2.0
        frames = []
        while offset + n < len(self.sample):
            frames.append(self.sample[offset:offset + n])
            timestamp += duration
            offset += n
        return frames


if __name__=='__main__':
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
            vader = vad_remove_silence(sample, 16000)
            vad_sample = vader.get_vad_sample()
            print vad_sample, len(vad_sample)

            ax1 = plt.subplot(211)
            ax1.set_title('Raw wave of ' + label)
            ax1.set_ylabel('Amplitude')
            ax1.plot(np.arange(0, len(sample), 1), sample)

            ax2 = plt.subplot(212)
            ax2.set_title('vad wave of ' + label)
            ax2.set_ylabel('vad litude')
            ax2.plot(np.arange(0, len(vad_sample), 1), vad_sample)

            plt.show()

    raw_input()



