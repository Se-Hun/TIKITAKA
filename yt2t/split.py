# -*- coding: utf-8 -*-
import pickle
import os
from random import shuffle

YT = "youtube2text_iccv15"
SHUFFLE = True

with open(os.path.join(YT, "CAP.pkl"), "rb") as f:
    ann = pickle.load(f, encoding="latin-1")
    #ann = pickle.load(f, encoding="utf-8")

vid2anns = {}
for vid_name, data in ann.items():
    for d in data:
        try:
            vid2anns[vid_name].append(d["tokenized"])
        except KeyError:
            vid2anns[vid_name] = [d["tokenized"]]

with open(os.path.join(YT, "train.pkl"), "rb") as f:
    train = pickle.load(f, encoding="latin-1")
    #train = pickle.load(f, encoding="utf-8")

with open(os.path.join(YT, "valid.pkl"), "rb") as f:
    val = pickle.load(f, encoding="latin-1")
    #val = pickle.load(f, encoding="utf-8")

with open(os.path.join(YT, "test.pkl"), "rb") as f:
    test = pickle.load(f, encoding="latin-1")
    #test = pickle.load(f, encoding="utf-8")

train_files = open("yt2t_train_files.txt", "w")
val_files = open("yt2t_val_files.txt", "w")
val_folded = open("yt2t_val_folded_files.txt", "w")
test_files = open("yt2t_test_files.txt", "w")

train_cap = open("yt2t_train_cap.txt", "w")
val_cap = open("yt2t_val_cap.txt", "w")

vid_names = vid2anns.keys()
if SHUFFLE:
    vid_names = list(vid_names)
    shuffle(vid_names)

for vid_name in vid_names:
    anns = vid2anns[vid_name]
    vid_path = vid_name + ".npy"
    for i, an in enumerate(anns):
        an = an.replace("\n", " ")  # some caps have newlines
        split_name = vid_name + "_" + str(i)
        if split_name in train:
            train_files.write(vid_path + "\n")
            train_cap.write(an + "\n")
        elif split_name in val:
            if i == 0:
                val_folded.write(vid_path + "\n")
            val_files.write(vid_path + "\n")
            val_cap.write(an + "\n")
        else:
            # Don't need to save out the test captions,
            # just the files. And, don't need to repeat
            # it for each caption
            assert split_name in test
            if i == 0:
                test_files.write(vid_path + "\n")
