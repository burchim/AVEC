import nnet
import os

# Params
workers_prepare = -1 # Set to -1 for nproc
mean_face_path = "media/20words_mean_face.npy"
tokenizer_path = "datasets/LRS3/tokenizerbpe256.model"
lrw_username = "" # Set to your lrw username
lrw_password = "" # Set to your lrw password
lrs2_username = "" # Set to your lrs2 username
lrs2_password = "" # Set to your lrs2 password
lrs3_username = "" # Set to your lrs3 username
lrs3_password = "" # Set to your lrs3 password

print("Download and Prepare LRW")
os.environ["LRW_USERNAME"] = lrw_username
os.environ["LRW_PASSWORD"] = lrw_password
lrw_dataset = nnet.datasets.LRW(None, None, download=True, prepare=True, mean_face_path=mean_face_path, workers_prepare=workers_prepare, mode="train")

print("Download and Prepare LRS2")
os.environ["LRS2_USERNAME"] = lrs2_username
os.environ["LRS2_PASSWORD"] = lrs2_password
lrs2_dataset = nnet.datasets.LRS(None, None, version="LRS2", download=True, prepare=True, tokenizer_path=tokenizer_path, mean_face_path=mean_face_path, workers_prepare=workers_prepare, mode="pretrain+train+val")

print("Download and Prepare LRS3")
os.environ["LRS3_USERNAME"] = lrs3_username
os.environ["LRS3_PASSWORD"] = lrs3_password
lrs3_dataset = nnet.datasets.LRS(None, None, version="LRS3", download=True, prepare=True, tokenizer_path=tokenizer_path, mean_face_path=mean_face_path, workers_prepare=workers_prepare, mode="pretrain+trainval")

print("Create Corpora")
lrs2_dataset.create_corpus(mode="pretrain")
lrs2_dataset.create_corpus(mode="train")
lrs2_dataset.create_corpus(mode="val")
lrs2_dataset.create_corpus(mode="test")
lrs3_dataset.create_corpus(mode="pretrain")
lrs3_dataset.create_corpus(mode="trainval")
lrs3_dataset.create_corpus(mode="test")

filenames = ["datasets/LRS2/corpus_pretrain.txt", "datasets/LRS2/corpus_train.txt", "datasets/LRS2/corpus_val.txt", "datasets/LRS3/corpus_pretrain.txt", "datasets/LRS3/corpus_trainval.txt"]
with open("datasets/LRS3/corpus_lrs23_pretrain+train+val.txt", "w") as fw:
    for filename in filenames:
        with open(filename, "r") as fr:
            for line in fr.readlines():
                fw.write(line)
