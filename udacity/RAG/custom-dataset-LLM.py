from openai import OpenAI
import os
import requests
from bs4 import BeautifulSoup

base_url = "https://openai.vocareum.com/v1"
api_key = os.environ.get("OPEN_AI_KEY")

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

# Making a GET request
response = requests.get('https://www.example.com')

# print the status code
print(response.status_code)

# print the content of the response
print(response.text)

# url = 'https://archive.org/details/cu31924067841738'
# response = requests.get(url)


# Opens (or creates) a file named language_of_flowers.html in write-binary mode (wb)
with open("language_of_flowers.html", mode='wb') as file:
    file.write(response.content)

with open("language_of_flowers.html") as fp:
    flower_soup = BeautifulSoup(fp, 'html.parser')

# Print a clean (prettier) version to look through
print(flower_soup.prettify())

book_title = flower_soup.find("title")
print(book_title)

book_title = book_title.text.strip()
print(book_title)

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "What is " + book_title,
        }
    ],
    model="gpt-4o",
)

print(response.choices[0].message.content)





