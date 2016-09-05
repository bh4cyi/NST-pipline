import re
fp = open("./target_text/sent.ascii.txt")
text = fp.read()
sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)


text_file = open("./target_text/Output.txt", "w")

for sentence in sentences:
    text_file.write("%s\n" % sentence.strip('\n').strip())

text_file.close()
fp.close()

text_file = open("./target_text/Output.txt")
target = open("./target_text/target.txt", "w")
for line in text_file:
    if re.match(r'^\s*$', line):
        print 'empty'
    else:
        target.write(line)
