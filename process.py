from selenium import webdriver
import pandas as pd
import requests as req

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument('disable-gpu')
driver = webdriver.Chrome("chromedriver", options=options)

df = pd.read_excel('data.xls')
df_list = df.values.tolist()
for ind, val in enumerate(df_list):
    uni_name = val[6]
    driver.get('https://www.uniprot.org/uniprot/?query=' + uni_name)
    driver.implicitly_wait(5)
    uni_id = driver.find_element_by_class_name("entryID").text
    res = req.post("http://www.uniprot.org/uniprot/" + uni_id + ".fasta")
    data = ''.join(res.text)
    with open("data/" + str(ind) + ".fasta", "w") as f:
        f.write(data)
driver.quit()
