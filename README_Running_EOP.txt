For running the EOP:

Follow the installation guide from https://github.com/hltfbk/EOP-1.2.3/wiki/Installation

The following step are ran for the different combinations of train and test data. The variables after trainFile and testFile can be altered to the train and testfile to your desire.
For our experiment, the same data was used and added to the same folder as the pre-delivered sets where found.

---Train---
java -Djava.ext.dirs=../EOP-1.2.3/ eu.excitementproject.eop.util.runner.EOPRunner -config ./eop-resources-1.2.3/configuration-files/MaxEntClassificationEDA_Base+OpenNLP_EN.xml -trainFile ./eop-resources-1.2.3/data-set/English_dev.xml 

---run testing set---
java -Djava.ext.dirs=../EOP-1.2.3/ eu.excitementproject.eop.util.runner.EOPRunner -config ./eop-resources-1.2.3/configuration-files/MaxEntClassificationEDA_Base+OpenNLP_EN.xml -test -testFile ./eop-resources-1.2.3/data-set/English_test.xml -output ./eop-resources-1.2.3/results/

---get scoring---
java -Djava.ext.dirs=../EOP-1.2.3/ eu.excitementproject.eop.util.runner.EOPRunner -testFile ./eop-resources-1.2.3/data-set/English_test.xml -score -results ./eop-resources-1.2.3/results/MaxEntClassificationEDA_Base+OpenNLP_EN.xml_results.txt

The scores then can be found in ./eop-resources-1.2.3/results/MaxEntClassificationEDA_Base+OpenNLP_EN.xml_results.txt_report