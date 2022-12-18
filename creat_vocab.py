from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import json
import torch
print(torch.cuda.is_available())

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])


tokenizer.pre_tokenizer = Whitespace()


files = ['QM.txt']
tokenizer.train(files, trainer)


tokenizer.save("QM.json")
tokenizer = Tokenizer.from_file("QM.json")

# infile = open('../fp_model_pre/100w.json', 'r')
# outfile = open('../fp_model_pre/vocab.txt', 'w')
#
# file = json.load(infile)
# vocab = file['model']['vocab']
#
# for label, feature in vocab.items():
#     outfile.write(label + '\n')
