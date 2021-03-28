#!/usr/bin/env sh
rsync -azv /home/frederic/Desktop/diffpanstarrsoutput2/*png ovh:/home/ubuntu/newimages
mv /home/frederic/Desktop/diffpanstarrsoutput2/*png /home/frederic/Desktop/diffpanstarrsoutput2_sent

