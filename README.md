# Create the environment
conda create -n nlpGroup python=3.7.1
conda activate nlpGroup

#Set up environment and install packages
conda install -c anaconda nltk
pip install scikit-multilearn
pip install vaderSentiment
