We provide an interface to ModelarDB's compression algorithms: ModelarDB.jar. 
Usage: 

``java -cp ModelarDB.jar ModelarDBRunner.java parquetFilePath errorBoundsInPercentages modelsToUse(C L G)``

* parquetFilePath is the url of the data to compress
* errorBoundsInPercentages used are 1, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 65, 80.   
* Models To Use:
  * C: runs PMC
  * L: runs SWING
  * G: runs GORILLA

