# Our Data So Far


## Conda Enviroment
Make the conda enviroment by running
```console
conda env create --file=environment.yml
```

This will get all the packages that we are using in the data science bootcamp and more that I have used.

## Downloading the data
When you are ready to download the data, you might need to get a kaggle account and get a token for your account and put it in ~/kaggle/kaggle.json on Linux, OSX, or other UNIX-based operating systems, and at C:\Users\<Windows-username>\.kaggle\kaggle.json on Windows.  If the token is not there a error will be raised.  To get the token, go to your account, setting, and API and "create New Token" and put it in the approprate place.

To download all the current data we have so far, and make the appropriate .db files, just run 

```console
init_get_data.py
```
in the main folder.


