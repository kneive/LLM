# custom search_images_bing() function from
# https://forums.fast.ai/t/getting-more-than-150-images-using-search-images-bing/77947/11

# pip install azure-cognitiveservices-search-imagesearch

import requests
from itertools import chain
from fastcore.foundation import L

#from azure.cognitiveservices.search.imagesearch import ImageSearchClient as api
#from msrest.authentication import CognitiveServicesCredentials as auth

def search_images_bing(key, term, total_count=150, min_sz=224):
    
    """Search for images using the Bing API
    
    :param key: Your Bing API key
    :type key: str
    :param term: The search term to search for
    :type term: str
    :param total_count: The total number of images you want to return (default is 150)
    :type total_count: int
    :param min_sz: the minimum height and width of the images to search for (default is 128)
    :type min_sz: int
    :returns: An L-collection of ImageObject
    :rtype: L
    """

    max_count = 150
    imgs = []
    for offset in range(0, total_count, max_count):
        count = max_count if total_count - offset > max_count else total_count - offset
        img = requests.get("https://api.bing.microsoft.com/v7.0/images/search", 
                            headers={"Ocp-Apim-Subscription-Key":key}, 
                            params={'q':term, 
                                    'min_height':min_sz, 
                                    'min_width':min_sz, 
                                    'count':count, 
                                    'offset':offset})
        results = img.json()
        imgs.append(L(results['value']))

    return L(chain(*imgs)).attrgot('contentUrl').unique()     
