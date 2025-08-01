import requests
from bs4 import BeautifulSoup
import os
import argparse 

base_link = 'https://sid.erda.dk/'
link = 'https://sid.erda.dk/cgi-sid/ls.py?share_id=f6hdp1zTzh&current_dir=.&flags=f'

def get_soup(link):
    source_code = requests.get(link)
    soup = BeautifulSoup(source_code.content,"lxml")
    f = []
    f.extend(soup.find_all('a', {'class' : ['leftpad directoryicon', ]}))
    f.extend(soup.find_all('a', {'title' : 'open'}))
    return f

# download file in link 
def download_file(link, destination):
    r = requests.get(link, allow_redirects=True)
    open(destination, 'wb').write(r.content)

def rec(link, destination = './'):
    f = get_soup(link)
    for child in f:
        if child.get('title') == 'open':
            link = f'{base_link}{child.get("href")}'
            child_path = child.get("href")[27:]
            os.makedirs(os.path.join(destination, os.path.dirname(child_path)), exist_ok=True)
            print(f'{destination}/{child_path}')
            download_file(link, f'{destination}/{child_path}')
        else:
            link = f'{base_link}cgi-sid/{child.get("href")}'
            rec(link, destination)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download files from a given link.')
    parser.add_argument('--output', type=str, default='.', help='Output directory to save the downloaded files.')
    args = parser.parse_args()

    print('DOWNLOADING')
    rec(link, destination = args.output)