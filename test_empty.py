import re
fp = open("./target_text/target.txt")
for line in fp:
    if re.match(r'^\s*$', line):
        print 'empty'
