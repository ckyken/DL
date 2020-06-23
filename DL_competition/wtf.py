import ipdb
import urllib.request
import urllib.error
import ssl


def download_stopwords(url):
    try:
        gcontext = ssl.SSLContext()
        response = urllib.request.urlopen(url, context=gcontext)
        data = response.read().decode('utf-8')
        return set(data.split('\n'))
    except Exception as e:
        print(e)
        return set()


stop_words = download_stopwords(
    'https://raw.githubusercontent.com/stopwords-iso/stopwords-zh/master/stopwords-zh.txt')
print(stop_words)
