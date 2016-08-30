import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
fp = open("./target_text/sent.ascii.txt")
data = fp.read()
data =  '\n'.join(tokenizer.tokenize(data)).encode('ascii','ignore')

text_file = open("Output.txt", "w")
text_file.write(data)
text_file.close()
