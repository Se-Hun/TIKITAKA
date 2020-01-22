import pickle
import os
import random
import string

seed = 2345
random.seed(seed)

YT = "youtube2text_iccv15"
SHUFFLE = True

with open(os.path.join(YT, "CAP.pkl"), "rb") as f:
    ann = pickle.load(f, encoding="latin-1")

def clean(caption):
    caption = caption.lower()
    caption = caption.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    # remove punctuation
    caption = caption.translate(str.maketrans("", "", string.punctuation))
    # multiple whitespace
    caption = " ".join(caption.split())
    return caption

with open(os.path.join(YT, "train.pkl"), "rb") as f:
    train = pickle.load(f, encoding="latin-1")

with open(os.path.join(YT, "valid.pkl"), "rb") as f:
    val = pickle.load(f, encoding="latin-1")

with open(os.path.join(YT, "test.pkl"), "rb") as f:
    test = pickle.load(f, encoding="latin-1")

train_data = []
val_data = []
test_data = []
for vid_name, data in ann.items():
    vid_path = vid_name + ".npy"
    for i, d in enumerate(data):
        split_name = vid_name + "_" + str(i)
        datum = (vid_path, i, clean(d["caption"]))
        if split_name in train:
            train_data.append(datum)
        elif split_name in val:
            val_data.append(datum)
        elif split_name in test:
            test_data.append(datum)
        else:
            assert False

if SHUFFLE:
    random.shuffle(train_data)

train_files = open("yt2t_train_files.txt", "w")
train_cap = open("yt2t_train_cap.txt", "w")

for vid_path, _, an in train_data:
    train_files.write(vid_path + "\n")
    train_cap.write(an + "\n")

train_files.close()
train_cap.close()

val_files = open("yt2t_val_files.txt", "w")
val_folded = open("yt2t_val_folded_files.txt", "w")
val_cap = open("yt2t_val_cap.txt", "w")

for vid_path, i, an in val_data:
    if i == 0:
        val_folded.write(vid_path + "\n")
    val_files.write(vid_path + "\n")
    val_cap.write(an + "\n")

val_files.close()
val_folded.close()
val_cap.close()

test_files = open("yt2t_test_files.txt", "w")

for vid_path, i, an in test_data:
    # Don't need to save out the test captions,
    # just the files. And, don't need to repeat
    # it for each caption
    if i == 0:
        test_files.write(vid_path + "\n")

test_files.close()
