class Map:

    def __init__(self, map_name):
        assert map_name=="Manhattan" or map_name=="Pittsburgh"
        self.graph=None

        #bounding boxes
        if map_name=="Manhattan":
            self.bounding_box_list=[Box(min_lat=40.695,min_lon=-74.028,max_lat=40.705,max_lon=-74),
                                    Box(min_lat=40.705,min_lon=-74.028,max_lat=40.712,max_lon=-73.975),
                                    Box(min_lat=40.712,min_lon=-74.028,max_lat=40.718,max_lon=-73.973),
                                    Box(min_lat=40.718,min_lon=-74.028,max_lat=40.725,max_lon=-73.972),
                                    Box(min_lat=40.725,min_lon=-74.02,max_lat=40.73,max_lon=-73.972),
                                    Box(min_lat=40.73,min_lon=-74.02,max_lat=40.738,max_lon=-73.969),
                                    Box(min_lat=40.738,min_lon=-74.02,max_lat=40.745,max_lon=-73.969),
                                    Box(min_lat=40.745,min_lon=-74.013,max_lat=40.752,max_lon=-73.963),
                                    Box(min_lat=40.752,min_lon=-74.013,max_lat=40.76,max_lon=-73.956),
                                    Box(min_lat=40.76,min_lon=-74.01,max_lat=40.7645,max_lon=-73.952),
                                    Box(min_lat=40.7645,min_lon=-74,max_lat=40.772,max_lon=-73.945)]
        else:
            self.bounding_box_list=[Box(min_lat=40.425,min_lon=-80.035,max_lat=40.460,max_lon=-80.01),
                        Box(min_lat=40.425,min_lon=-80.01,max_lat=40.460,max_lon=-80),
                        Box(min_lat=40.425,min_lon=-80,max_lat=40.460,max_lon=-79.98),
                        Box(min_lat=40.425,min_lon=-79.98,max_lat=40.460,max_lon=-79.955),
                        Box(min_lat=40.425,min_lon=-79.955,max_lat=40.460,max_lon=-79.93)]
                        

        entities=self.__get_all_entities()
    
    def __get_all_entities (self):
        osm_items=[]
        for bounding_box in self.bounding_box_list:
            list_dic_osm=self.__get_entities_by_geo_box(bounding_box) 
            osm_items=self.__osm_filter(list_dic_osm,osm_items)


    def __osm_filter(list_dic_osm,osm_items):
        for item in list_dic_osm:
            if item['type'] == 'relation':
                continue
            



    def __get_entities_by_geo_box(self, box):
        api = osmapi.OsmApi()

        for i in range (3): #OSM limmits the number of requests so we might need to wait
            try:
                list_dic_osm = api.Map(min_lat=box.min_lat, min_lon=box.min_lon, max_lat=box.max_lat, max_lon=box.max_lon)
                return list_dic_osm
            except:
                time.sleep(60)
        
        return None