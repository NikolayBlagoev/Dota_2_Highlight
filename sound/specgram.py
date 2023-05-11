N = 256
w = np.hamming(N)
ov = N - int(a.frame_rate/30)
print(a.frame_rate)
Pxx, freqs, bins = mlab.specgram(y, NFFT=N, Fs=int(a.frame_rate/30), 
                                 noverlap=ov, mode="magnitude")

print(np.log10(Pxx).shape)
transposed = np.transpose(np.log10(Pxx))
print(transposed.shape)
print(freqs.shape)

n1, n2 = int(23*20/a.duration_seconds*len(bins)), int(24*20/a.duration_seconds*len(bins))
print(n1, n2)
ex1 = bins[n1], bins[n2], freqs[0], freqs[-1]
pyplot.imshow(np.flipud(np.log10(Pxx)), extent=ex1)
pyplot.axis('auto')
pyplot.axis(ex1)
pyplot.xlabel('time (s)')
pyplot.ylabel('freq (Hz)')
pyplot.savefig("test.png")
pyplot.show()