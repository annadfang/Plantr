# import the necessary packages
from azure.cognitiveservices.search.imagesearch import ImageSearchAPI
from msrest.authentication import CognitiveServicesCredentials
import numpy as np
import urllib.request
import cv2
import os


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


searchs = ["money tree plant",
           "spider plant",
           "aloe vera plant",
           "peace lily",
           "arrowhead plant",
           "snake plant",
           "boston fern",
           "croton plant",
           "elephants ear plant",
           "caladium plant",
           "lucky bamboo",
           "parlor palm",
           "sago palm",
           "venus fly trap",
           "jade plant",
           "flaming katy succulent",
           "poinsettia plant",
           "golden pothos plant",
           "weeping fig plant",
           "english ivy plant"]

files = ["money_tree",
         "spider_plant",
         "aloe_vera",
         "peace_lily",
         "arrowhead_plant",
         "snake_plant",
         "boston_fern",
         "croton_plant",
         "elephants_ear_plant",
         "caladium_plant",
         "lucky_bamboo",
         "parlor_palm",
         "sago_palm",
         "venus_fly_trap",
         "jade_plant",
         "flaming_katy_succulent",
         "poinsettia_plant",
         "golden_pothos_plant",
         "weeping_fig_plant",
         "english_ivy_plant"]

# Bing Search API information
subscription_key = "83c3bc324d1f4af8bbe0a0f8532de789"

k = 0
search_term = searchs[k]

"""
      This application will search images on the web with the Bing Image Search
      API and print out first image result.
      """
# create the image search client
client = ImageSearchAPI(CognitiveServicesCredentials(subscription_key))

# send a search query to the Bing Image Search API
image_results = client.images.search(query=search_term, count=100)
print("Searching the web for images of: {}".format(search_term))

# Image results
if image_results.value:
    print("Total number of images returned: {}".format(
        len(image_results.value)))
else:
    print("Couldn't find image results!")

# loop over the image URLs
count = 1
i = 0
while count < 50:
    # download the image URL and display it
    try:
        urllib.request.urlretrieve(
            image_results.value[i].content_url,
            os.getcwd() + "/plant_imgs/" + files[k] + "/img" + str(count) + ".jpg")
        count += 1
    except urllib.error.HTTPError as error:
        print(error)
    except urllib.error.URLError as e:
        print(e)
    i += 1
