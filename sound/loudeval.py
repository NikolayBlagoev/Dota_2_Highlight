from matplotlib import pyplot
from os import path
import soundfile as sf
import pyloudnorm as pyln

DATA_DIR    = "data"
SOUND_FILE  = "mono.mp3"

if __name__ == "__main__":
    # Load data and compute loudness over time
    data_path   = path.join(DATA_DIR, SOUND_FILE)
    data, rate  = sf.read(data_path)
    meter       = pyln.Meter(rate)
    lodness_over_time = []
    for i in range(int(len(data)/rate)):
        loudness = meter.integrated_loudness(data[int(i * rate ) : int((i + 1) * rate)])
        lodness_over_time.append(loudness)

    # Plot loudness over time
    pyplot.plot(lodness_over_time)
    pyplot.savefig("results/loudness_pyloud.png")
    pyplot.show()
