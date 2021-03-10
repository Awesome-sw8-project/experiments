import json



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


sensordata = [
    {
        "timestamp": 1578475327694, 
        "Type" : "TYPE_WIFI",
        "ssid" : "d839a45ebe64ab48b60a407d837fb01d3c0dfef9",
        "bssid" : "edfec57e7fdec0f9decbe1e65d7e04539d4e68ba",
        "RSSI" : -46,
        "frequency" : 5180
    },
    {
        "timestamp": 1578475327694, 
        "Type" : "TYPE_WIFI",
        "ssid" : "da39a3ee5e6b4b0d3255bfef95601890afd80709",
        "bssid" : "e7d431b2031b1f47e69f73d608d11422838b72fa",
        "RSSI" : -46,
        "frequency" : 5180
    },
    {
        "timestamp": 1578475327694, 
        "Type" : "TYPE_WIFI",
        "ssid" : "b6ffe5619e02871fcd04f61c9bb4b5c53a3f46b7",
        "bssid" : "61266592622383e219e7fb5f40374fc27552ad12",
        "RSSI" : -47,
        "frequency" :5180
    },
    {
        "timestamp": 1578475327694, 
        "Type" : "TYPE_WIFI",
        "ssid" : "d839a45ebe64ab48b60a407d837fb01d3c0dfef9",
        "bssid" : "edfec57e7fdec0f9decbe1e65d7e04539d4e68ba",
        "RSSI" : -46,
        "frequency" :5180
    },
    {
        "timestamp": 1578475327694, 
        "Type" : "TYPE_WIFI",
        "ssid" : "da39a3ee5e6b4b0d3255bfef95601890afd80709",
        "bssid" : "e7d431b2031b1f47e69f73d608d11422838b72fa",
        "RSSI" : -46,
        "frequency" : 5180
    },
    {
        "timestamp": 1578475327694, 
        "Type" : "TYPE_WIFI",
        "ssid" : "b6ffe5619e02871fcd04f61c9bb4b5c53a3f46b7",
        "bssid" : "61266592622383e219e7fb5f40374fc27552ad12",
        "RSSI" : -47,
        "frequency" :5180
    }
]
construct_data("5a0546857ecc773753327266", "5e15a2c4f4c3420006d521cb", sensordata)