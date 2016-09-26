import re
fp = open("./target_text/sent.ascii.txt")
for line in fp:
    if re.match(r'^\s*$', line):
        print 'empty'
