import re
fp = open("./target_text/sent.ascii.txt")
text = fp.read()
sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
print sentences[1]
"""
text_file = open("./target_text/Output.txt", "w")
for sentence in sentences:
    if re.match(r'^\s*$', sentence):
        print 'empty'
    else:
        text_file.write("%s\n" % sentence)

text_file.close()
fp.close()
"""
