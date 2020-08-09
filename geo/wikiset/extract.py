import requests
import shapely
from shapely.geometry.point import Point
import shapely.wkt



place_polygon = shapely.wkt.loads(
    'POLYGON ((-73.9455846946375 40.7711351085905,-73.9841893202025 40.7873649535321,-73.9976322499213 40.7733311258037,-74.0035177432988 40.7642404854275,-74.0097394992375 40.7563218869601,-74.01237903206 40.741427380319,-74.0159612551762 40.7237048027967,-74.0199205544099 40.7110727528606,-74.0203570671504 40.7073623945662,-74.0188292725586 40.7010329598287,-74.0087894795267 40.7003781907179,-73.9976584046436 40.707144138196,-73.9767057930988 40.7104179837498,-73.9695033328803 40.730061057073,-73.9736502039152 40.7366087481808,-73.968412051029 40.7433746956588,-73.968412051029 40.7433746956588,-73.9455846946375 40.7711351085905))'
)


def get_wikidata_id_from_wikipedia_id(ID):

    url = (
        'https://en.wikipedia.org/w/api.php'
        '?action=query'
        '&prop=pageprops'
        '&ppprop=wikibase_item'
        '&redirects=1'
        f'&pageids={ID}'
        '&format=json'
        )
    json_response = requests.get(url).json()
    pages = json_response['query']['pages']
    wikidata_ids=[]
    for key, page in pages.items():
        wikidata_ids.append(page['pageprops']['wikibase_item'])
    return wikidata_ids

def get_wikipedia_by_geosearch(longitude,latitude,radius,limit=600):


    url = (
        'https://en.wikipedia.org/w/api.php'
        '?action=query'
        '&list=geosearch'
        f'&gscoord={latitude}|{longitude}'
        f'&gsradius={radius}'
        f'&gslimit={limit}'
        '&format=json'
        )
    json_response = requests.get(url).json()
    pages = json_response['query']['geosearch']

    return pages

items_all=[]
add_lon=0
add_lat=0

for j in range(50):
    for i in range(50):
        items=get_wikipedia_by_geosearch(latitude=40.695+add_lat, longitude=-74.028+add_lon ,radius=1000)
        add_lon+=0.001
        assert len(items)<500
        for item in items:
            if item not in items_all:
                items_all.append(item)
    add_lat += 0.001


min_lat=40.695
min_lon=-74.028
max_lat=40.788
max_lon=-73.940




all_ids=[]
wikipedia_found=[]
for idx, item in enumerate(items_all):
    # print(idx)
    try:
        x = item['lon']
        y = item['lat']
        point=Point(x,y)
        
        if not place_polygon.contains(point):
            continue
        if item['pageid'] not in wikipedia_found:
            wikipedia_found.append(item['pageid'])
        wikdata_ids=get_wikidata_id_from_wikipedia_id(item['pageid'])
        if wikdata_ids[0] not in all_ids:
            all_ids.append(wikdata_ids[0])
            print ('wd:{}'.format(wikdata_ids[0], item))
    except:

        pass
print (len(list(set(all_ids))))
print ('wiki:',len(wikipedia_found))
print ("END")

