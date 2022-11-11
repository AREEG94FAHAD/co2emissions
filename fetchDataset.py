import urllib.request


def fechtheData(path, filename):
    # path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
    urllib.request.urlretrieve(path, filename)

