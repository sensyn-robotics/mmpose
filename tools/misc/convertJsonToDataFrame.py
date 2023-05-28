import pandas as pd
import json
import errno
import os
import json



def convertJsonToDf(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), filename)

    with open(filename, 'r') as myJson:
        results = json.load(myJson)

    dfs = []

    for result in results:
        df = pd.DataFrame.from_dict(result)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index= True)

    return df

if __name__=="__main__":
    
    df = convertJsonToDf("test.json")
    print(df)



