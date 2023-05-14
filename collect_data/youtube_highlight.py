import requests
from matplotlib import pyplot as plt
from sys import argv
response = requests.get(f"https://yt.lemnoslife.com/videos?part=mostReplayed&id={argv[1]}")
heatmap = response.json()["items"][0]["mostReplayed"]["heatMarkers"]
arr_excitation = []
for t in heatmap:
    arr_excitation.append(t["heatMarkerRenderer"]["heatMarkerIntensityScoreNormalized"])
plt.plot(arr_excitation)
plt.savefig(f"{argv[2]}/youtube_excitation.png")
plt.show()
