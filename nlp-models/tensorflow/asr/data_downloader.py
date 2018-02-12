from urllib.request import urlopen, urlretrieve
from bs4 import BeautifulSoup

import os
import re
import pprint


prefix = 'https://tspace.library.utoronto.ca'
dir = './data/'

base_url = 'https://tspace.library.utoronto.ca/handle/1807/24'
urls = [base_url+str(i) for i in range(488, 502)]
#pprint.pprint(urls)

count = 1
for url in urls:
    soup = BeautifulSoup(urlopen(url).read(), 'html5lib')
    for a in soup.findAll('a', href=re.compile(r'/bitstream/.*.wav')):
        link = a['href']
        print(count, a['href'])
        
        audio_save_loc = dir+link.split('/')[-1]
        if os.path.isfile(audio_save_loc):
            print("File Already Exists")
        urlretrieve(prefix+a['href'], audio_save_loc)

        with open(audio_save_loc.replace('.wav', '.txt'), 'w') as f:
            f.write('say the word ' + link.split('_')[-2])
        
        count += 1

"""
with open(audio_save_loc.replace('.wav', '.txt')) as f:
    print(f.read())
"""
