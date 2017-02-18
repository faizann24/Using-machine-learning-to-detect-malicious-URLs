# Extract malicios URLS from a MISP CSV dump
# and write it to a new dataset file
#
#

if2 = open("data2.csv","a")
if1 = open("misp.csv_sig.JShome.csv", "rb")
for line in if1.readlines():
    spline = str(line).split(",")
    try:
        if spline[2] in ["Network activity", "Payload delivery"]:
            if spline[3] in ["url", "uri"]:
                item = spline[4].strip('''"''')
                if item.startswith("/"):
                    continue
                elif item.startswith("https://"):
                    item = item[8:]
                elif item.startswith("http://"):
                    item = item[7:]
		print(item)
                if2.write("%s,%s\n" % (item, "bad"))
    except:
       continue
if2.close()
