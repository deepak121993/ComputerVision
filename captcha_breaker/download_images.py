

import argparse
import request
import time 
import os



###this script is for downloading the data froom the url and storing those captha files in local disk.

ap = argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True)
ap.add_argument("-n","--num-images",required=True)
args = vars(ap.parse_args())

url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total=0

for i in range(0,args["num-images"]):

    try:
        r = request.get(url,timeout=60)

        #save 
        p = os.path.sep.join(args["output"],"{}.jpg".format(str(total).zfill(5)))

        f= open(p,"wb")
        f.write(r.content)
        f.close()
        total += 1

    except:
        print("[INFO] got an error bro , look into this")
    

    time.sleep(0.1)


