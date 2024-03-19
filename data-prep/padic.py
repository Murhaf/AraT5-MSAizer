"""
Assuming the PADIC dataset is in the data folder, this script will convert the XML file to a JSON file
and then upload it to the Hugging Face Hub.
"""

import random
import xml.etree.ElementTree as ET

import pandas as pd
from datasets import load_dataset


with open('data/PADIC-20-02-2017.xml', 'r') as file:
	xml_string = file.read()

root = ET.fromstring(xml_string)



# Extract data from XML and create a list of dictionaries
data = []
for sentence in root.findall('Sentence'):
    msa_text = sentence.find('MSA').text
    for lang in ['ALG', 'ANB', 'TUN', 'PAL', 'SYR', 'MAR']:
        lang_text = sentence.find(lang).text
        data.append({'msa': msa_text, 'Language': lang, 'dialect': lang_text})

    if random.choice([True, False]):
        data.append({'msa': msa_text, 'Language': 'msa', 'dialect': msa_text})        

# Create DataFrame
df = pd.DataFrame(data)


df.to_json(
    "padic.json",
    orient="records",
    lines=True,
    force_ascii=False,
)

dataset = load_dataset("json", data_files="padic.json")
dataset.push_to_hub("padic_2017", private=True)
