import pickle
import os

YT = "youtube2text_iccv15"
YTC = "YouTubeClips"

# load the YouTube hash -> vid### map.
with open(os.path.join(YT, "dict_youtube_mapping.pkl"), "rb") as f:
    yt2vid = pickle.load(f, encoding="latin-1")

for f in os.listdir(YTC):
    hashy, ext = os.path.splitext(f)
    vid = yt2vid[hashy]
    fpath_old = os.path.join(YTC, f)
    f_new = vid + ext
    fpath_new = os.path.join(YTC, f_new)
    os.rename(fpath_old, fpath_new)
