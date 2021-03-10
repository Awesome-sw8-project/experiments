import json

data = {
    "siteID" : 123,
    "siteName" : "nameOfSite",
    "floorLevel" : 0,
    "pathID" : 1232324,
    "startTime" : 123142141,
    sensorData : [
        {},
    ]
}

def construct_data(siteID, pathID, sensorData):
    data = {
        "siteID" : siteID,
        "siteName" : "nameOfSite",
        "floorLevel" : 0,
        "pathID" : pathID,
        "startTime" : 123142141,
        "sensorData" : sensorData
    }
    
    with open(str(siteID)+'_'+str(pathID)+'.json', 'w') as f:
        json.dump(data,f)


