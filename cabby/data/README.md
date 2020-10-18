# WikiGeo - a dataset to support the geolocation subtask

The goal of the RVS task is to identify the goal location. From that perspective it is a geolocation. To support the geolocation identification we divide the region into S2Cell ids and predict and aim to predict as small an S2Cell as possible that contains the goal location. To correctly predict S2Cells we haven’t encountered before we want to learn an S2Cell embedding model. To do that we collect an additional dataset, one which contains descriptions of locations and their geolocation point. We collect the dataset from three different sources: (1) Wikipedia, (2) Wikidata, (2) OSM. There are four different types of samples: (1) Wikipedia pages; (2) Wikipedia backlinks; (3) Wikidata pages; (4) OSM entities.

To create a Wikipedia pages based sample, we first extract Wikidata items located in the region. Every Wikidata page has a corresponding Wikipedia page. We query the Wikimedia API and extract the Wikipedia corresponding pages. For each sentence in the Wikipedia page we construct a sample. E.g., "The club plays in the Eastern Conference of the USL Championship, the second tier of the American soccer pyramid."

To create a Wikipedia backlinks sample we use the Wikipedia pages extracted.  We query Wikimedia API for Wikipedia backlinks to the Wikipedia page extracted in 1. The backlinks are Wikipedia  pages that have some link to the Wikipedia pages queried on. For each sentence in the backlink with a mention of the entity queried on we extract it and construct a WIkiGeo sample. E.g., "The building itself is considered a Chinatown 'town hall'."

To create a Wikidata-based sample we query Wikidata by region and extract properties found: 
1. The place name
2. The place description 
3. Architectural style 
4. Use (e.g., hotel)
5. Has part (e.g., Knoxville Branch Library is part of Carnegie Library of Pittsburgh) 
6. Subsidiary (e.g., the subsidiary of CMU is Carnegie Mellon Silicon Valley)
7. Alternative name
8. Instance (e.g., Hillman Library is an instance of a  library)

We concatenate all these tags into one semi-sentence which is one sample in the WikiGeo dataset.  E.g, “Heinz Field and American football stadium that is home to the Pittsburgh Steelers and Pittsburgh Panthers and Steelers Stadium and stadium.”

To create an OSM-based sample we query OSM and extract POI that are not large (spread over less than 10 s2cellids of level 18.), within region, and have the following tag keys: 
1. 'name'
2. 'amenity' 
3. 'colour'
4. 'brand'
5. 'tourism'
6. 'leisure'
7. 'historic'
8. 'building'
9. 'description'
10. 'building:colour'
11. 'building:material'
12. 'roof:material'
13. 'roof:shape'
14. 'roof:colour'

Like the sample Wikidata-based samples, we concatenate the tag values into one semi-sentence. E.g,  “St. Paul Cathedral Rectory and building and Dithridge Street”.
