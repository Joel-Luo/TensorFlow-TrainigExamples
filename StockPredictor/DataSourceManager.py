import pandas as pd

class DataSourceManager():
    """ Data Source Manager """
    def __init__(self):
        return
    # End def 

    # Get Data from CSV file,
    # <Param>string path</Param>
    @classmethod
    def GetDataSourceFromCSV(self, path) -> pd.DataFrame:
        return pd.read_csv(path)
    # End def

# End class