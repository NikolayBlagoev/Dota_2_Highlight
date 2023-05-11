import soundfile as sf
import pyloudnorm as pyln
from matplotlib import mlab, pyplot
data, rate = sf.read("data/mono.mp3")
meter = pyln.Meter(rate)
lodness_over_time = []
for i in range(int(len(data)/rate)):
    loudness = meter.integrated_loudness(data[int(i * rate ) : int((i + 1) * rate)])
    lodness_over_time.append(loudness)

pyplot.plot(lodness_over_time)
pyplot.savefig("results/loudness_pyloud.png")
pyplot.show()