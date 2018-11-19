import pandas as pd
from bs4 import BeautifulSoup
import json

# load and filter data
data = pd.read_csv('wp_posts.csv', sep=';')
data = data[(data['post_type'] == 'post') & (data['post_status'] == 'publish')]
data = data[['post_content']].reset_index(drop=True)

# parse html
for i in data.index:
    soup = BeautifulSoup(data['post_content'].loc[i], 'html.parser')
    data['post_content'].loc[i] = soup.get_text()
    data['post_content'].loc[i] = data['post_content'].loc[i].lower()
    
# remove end-of-post links
data['post_content_clean'] = ""
for i, post in enumerate(data['post_content']):
    if "read more" in post:
        post_split = post.split("read more")
        data['post_content_clean'][i] = post_split[0]
    elif "further reading" in post:
        post_split = post.split("further reading")
        data['post_content_clean'][i] = post_split[0]
    elif "continue reading" in post:
        post_split = post.split("continue reading")
        data['post_content_clean'][i] = post_split[0]
    else:
        data['post_content_clean'][i] = post
        
# replace rare characters
for char in ['$', '%', '#', '&', '/', '@', '<', '>', '=', '[', ']', '\\', '_', '|', '£', '§',
              '©', '°', '¼', 'ß', 'à', 'á', 'ä', 'æ', 'è', 'é', 'í', 'ó', 'ö', 'ø', 'ü', 'ę', 'ō', 'δ',
              '†', '…', '↑', '→', '↓', '⇒', '∅', '≠', '(', ')', '∩', '+', '“', '”', '"']:
    data['post_content_clean'] = data['post_content_clean'].str.replace(char, '')
for char in ['’', '‘']:
    data['post_content_clean'] = data['post_content_clean'].str.replace(char, "'")
for char in ['–', '—', '-', '−']:
    data['post_content_clean'] = data['post_content_clean'].str.replace(char, "-")
data['post_content_clean'] = data['post_content_clean'].str.replace('\xa0', " ")

# convert content into list and save it as json file
text = []
for i in data['post_content_clean']:
    text.append(i)
with open("blog_posts_raw.json", "w") as file:
    json.dump(text, file)

# convert content into string and save it as text file
text = ""
for i in data['post_content_clean']:
    text = text + i + " "
with open("blog_posts_raw.txt", "w") as file:
    file.write(text)
