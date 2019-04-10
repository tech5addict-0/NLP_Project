# NLP Paper Reproduction 
## ["Emergent: a novel data-set for stance classification"](https://www.aclweb.org/anthology/N16-1138)

Implementation for Group 14 -  IN4325.

Team members:
 * [Priya Sarkar](https://www.linkedin.com/in/priyasarka)
 * [Nathan Buskulic](https://www.tudelft.nl/ewi/)
 * [Marjolein Knijff](https://www.tudelft.nl/ewi/)
 * [Zhang](https://www.tudelft.nl/ewi/)
 
# Python Version 
python=3.7.2

# Create the environment
```
conda create -n nlpGroup python=3.7.2
conda activate nlpGroup
```

# Set up environment and install packages
```
pip install scikit-multilearn
pip install stanfordnlp
pip install nltk
```

# Run 
```
python main.py 
```
You need to obtain [Emergent dataset](https://
github.com/willferreira/mscproject), [PPDB 2.0 database](http://paraphrase.org/#/) and the [Google news Dataset](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) as described in the paper and copy it to data folder and run the main.py file
