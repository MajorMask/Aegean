conda create -n Day1 python==3.9 opencv matplotlib numpy
conda init
conda activate Day1

# For HSV in action using CV 
python3 hsv.py

# For visualization
python3 -m http.server 8080  # 8081 or whichever one is available

# You can just rename this file to setup.bash or setup.sh and go to the terminal to activate this file ./setup.bash or ./setup.sh or bash setup.sh
