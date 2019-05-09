import requests
from bs4 import BeautifulSoup
import json

fifa_url = "http://www.fifa.com/worldcup/preliminaries/index.html"
fifa_html = requests.get(fifa_url).content

# Get National selections classified to Russia 2018
soup = BeautifulSoup(fifa_html, "lxml")
teams_html = soup.find_all("a", {"class": "team"})
list_teams = {}
for i in teams_html:
    idx = i['href'].find("team=")
    country_name = i.img['alt']
    country_id = i['href'][idx+5:idx+10]
    list_teams[country_name] = []

num_matches = 0
last_year = 2017
month = 12

matches_url1 = "http://www.fifa.com/worldcup/preliminaries/matches/year="
matches_url2 = "/month="
matches_url3 = "/index.htmx"

# Get matches of each national selection
while last_year >= 2015:

    matches_url_complete = matches_url1 + str(last_year) + matches_url2 + str(month) + matches_url3

    matches_html = requests.get(matches_url_complete).content
    soup = BeautifulSoup(matches_html, "lxml")
    current_list_matches = soup.find_all("div", {"class": "col-xs-12 clear-grid "})

    if len(current_list_matches) > 0:
        idx = str(current_list_matches[0].div.div)
        if idx.find("//img.fifa.com/images/layout/error/404.png") == -1:

            print("Extracting data from " + str(last_year) + "/" + str(month))

            for match in current_list_matches:
                if match.div.a is not None:
                    if match.div.a.div is not None:
                        data_match = match.div.a.find_all('div')

                        date = data_match[1].text
                        stadium = data_match[7].text
                        city = data_match[8].text
                        local = data_match[13].span.text
                        visitant = data_match[16].span.text
                        score = data_match[21].span.text

                        dict_match = {}

                        dict_match['date'] = date
                        dict_match['stadium'] = stadium
                        dict_match['city'] = city
                        dict_match['local'] = local
                        dict_match['visitant'] = visitant
                        dict_match['score'] = score
                        dict_match['islocal'] = True
                        print(dict_match)
                        if local in list_teams:
                            list_teams[local].append(dict_match)
                        if visitant in list_teams:
                            dict_match['islocal'] = False
                            list_teams[visitant].append(dict_match)
            print()
    if month > 1:
        month -= 1
    else:
        month = 12
        last_year -= 1

print()
print("Saving in russia2018.json...")
dict_out = {}
dict_out["teams"] = []
for country, matches in list_teams.items():
    dict_act = {}
    dict_act["contry"] = country
    dict_act["matches"] = matches[::-1]
    dict_out["teams"].append(dict_act)

with open('russia2018.json', 'w') as file:
    json.dump(dict_out, file)
print("Finished!")
