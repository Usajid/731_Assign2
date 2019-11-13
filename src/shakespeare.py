#!/usr/bin/env python
# coding: utf-8

# # To Be or Not To Be (EECS-731- Assignment # 02)
# ---

# In[16]:


get_ipython().run_cell_magic('time', '', '\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport pandas as pd\nimport numpy as np\n\n# modules for classification\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.naive_bayes import MultinomialNB\nfrom sklearn.svm import LinearSVC\n\n#sns.set (color_codes = True)\npd.options.mode.chained_assignment = None\n\n# Render out plots inline\n%matplotlib inline')


# ## Dataset Loading into Pandas Frame and Viewing

# In[2]:


# Load the dataset into pandas
shakespeare = pd.read_csv ("../data/Shakespeare_data.csv")
print ("# Shakespear Dataset Dimensions: ", shakespeare.shape)
shakespeare.head()


# ## Dataset Cleaning for Empty or NaN values (dropna)

# In[17]:


# Removing rows with empty or NaN fields
cl_shakespeare = shakespeare.dropna ()
print ("#Shakespear Dataset Dimensions (Without Missing Values): ", cl_shakespeare.shape)
cl_shakespeare.head ()


# ## Feature Extraction

# In[19]:


# Splitting 'ActSceneLine' into 3 columns to have more features to play with
cl2_shake = cl_shakespeare.drop (['ActSceneLine', 'Dataline', 'PlayerLinenumber'], axis = 1)
cl2_shake.tail()


# ## A Unique Players Gist

# In[5]:


cl2_shake ["Player"].unique ()[:50]


# ## Data Distribution

# In[6]:


player_refs = cl2_shake ["Player"].value_counts ()
player_refs_mean = int (round (np.mean (player_refs)))
player_refs_median = int (np.median (player_refs))
print_limits = 5

print ("# Top-%d Most Referenced Players\n" % print_limits, player_refs [:print_limits])
print ("\n## Bottom-%d Most Referenced Players\n" % print_limits, player_refs [-print_limits:])
print ("\nAverage Player Reference Count: ", player_refs_mean)
print ("Median  Player Reference Count: ", player_refs_median)

pl_dist = sns.distplot (player_refs, rug = 'True', rug_kws = {"color": "skyblue"})
pl_dist.set_xlabel ('Players', fontweight = 'bold')
pl_dist.set_ylabel ('Probability Density Function', fontweight = 'bold')
pl_dist.set_yticks ([])
pl_dist.set_ylim ([-0.0005, 0.015])
pl_dist.set_xlim ([0, 2000])
pl_dist.set_title ("Player Reference Counts Distribution across the Shakespeare's Plays", fontweight = 'bold')
pl_dist.title.set_position([.5, 1.05])


# ## Dataset Balancing

# In[7]:


ply_shake = cl2_shake.groupby ('Player')
filtered_shake = ply_shake.apply (lambda x: x.sample (player_refs_mean, replace = True)).reset_index (drop = True).drop_duplicates ('PlayerLine')
filtered_shake ['PlayerID'] = filtered_shake ['Player'].factorize ()[0]
print ("# Data Shape after Filtering: ", filtered_shake.shape)
print ("# A Snapshot of Filtered Data")
filtered_shake.tail ()


# ## Balanced Data Visualization

# In[8]:


flt_player_refs = filtered_shake ["Player"].value_counts ()
flt_player_refs_mean = int (round (np.mean (player_refs)))
flt_player_refs_median = int (np.median (player_refs))

print ("# Top-%d Most Referenced Players\n" % print_limits, flt_player_refs [:print_limits])
print ("\n## Bottom-%d Most Referenced Players\n" % print_limits, flt_player_refs [-print_limits:])
print ("\nAverage Player Reference Count: ", flt_player_refs_mean)
print ("Median  Player Reference Count: ", flt_player_refs_median)

pl_dist = sns.distplot (flt_player_refs, rug = 'True', rug_kws = {"color": "skyblue"})
pl_dist.set_xlabel ('Players', fontweight = 'bold')
pl_dist.set_ylabel ('Probability Density Function', fontweight = 'bold')
pl_dist.set_yticks ([])
pl_dist.set_ylim ([0, 0.025])
pl_dist.set_xlim ([0, player_refs_mean])
pl_dist.set_title ("Filtered Player Reference Counts Distribution across Shakespeare's Plays", fontweight = 'bold')
pl_dist.title.set_position([.5, 1.05])


# ## Player Names to IDs and Vice-Versa (Dictionary)

# In[9]:


player_id_df = filtered_shake [['Player', 'PlayerID']].drop_duplicates ().sort_values ('PlayerID')
player_to_id = dict (player_id_df.values)
id_to_player = dict (player_id_df [['PlayerID', 'Player']].values)
list(id_to_player.items())[:print_limits]


# ## Feature Transformation

# In[10]:


tfidf = TfidfVectorizer (sublinear_tf = True, min_df = 5, norm = 'l2', ngram_range = (1, 2), stop_words = 'english')
features = tfidf.fit_transform (filtered_shake.PlayerLine + filtered_shake.Play).toarray ()
labels = filtered_shake.PlayerID
features.shape


# ##  Dataset Training and Testing based Splitting

# In[11]:


X_train, X_test, y_train, y_test = train_test_split (features, labels, test_size = 0.25, random_state = 0)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# # CLASSIFIERS
# Next, we train/test different classifiers and investigate their accuracy.

# ## Training and Testing Classifer # 01: Multinomial Naive Bayes

# In[21]:


mnb = MultinomialNB ().fit (features, labels)


# In[22]:


print ('Multinomial Naive Bayes Accuracy Score:', round (mnb.score (X_test, y_test), 3))


# In[17]:


mnb_predict = mnb.predict (X_test)
pl_mnb = plt.scatter (y_test, mnb_predict)
plt.xlabel('True Values')
plt.ylabel('Predictions')


# ## Training and Testing Classifer # 02: Linear SVC (Without any Kernel)

# In[33]:


get_ipython().run_cell_magic('time', '', 'svc = LinearSVC ().fit (features, labels)')


# In[26]:


print ('Linear SVC Score:', round (svc.score (X_test, y_test), 3))


# In[34]:


svc_predict = svc.predict (X_test)
pl_svc = plt.scatter (y_test, svc_predict)
plt.xlabel('True Values')
plt.ylabel('Predictions')


# ## Training and Testing Classifer # 03: Logistic Regression

# In[35]:


get_ipython().run_cell_magic('time', '', 'lgr = LogisticRegression ().fit (features, labels)')


# In[38]:


print ('Logistic Regression based Score:', round (lgr.score (X_test, y_test), 3))


# In[39]:


lgr_predict = lgr.predict (X_test)
pl_lgr = plt.scatter (y_test, lgr_predict)
plt.xlabel('True Values')
plt.ylabel('Predictions')


# ## Training and Testing Classifer # 04: Random Forest

# In[12]:


get_ipython().run_cell_magic('time', '', 'rfc = RandomForestClassifier (random_state = 0).fit(features, labels)')


# In[13]:


print ('Random Forest  Classifier Accuracy Score:', round (rfc.score (X_test, y_test), 3))


# In[14]:


rfc_predict = rfc.predict (X_test)
pl_rfc = plt.scatter (y_test, rfc_predict)
plt.xlabel('True Values')
plt.ylabel('Predictions')


# # Conclusion

# First, we pre-processed and balanced the given Shakespeare dataset. Then, we analyzed four major machine learning classifiers to classify correct Players label. Out of four tested classifiers, Random Forest based classifier outperformed other methods with best accuracy of 93.7%. One reason of Random Forest, being the best, is their property of ensembleness as final classifiation score is not based on just one Decision Tree, but from several Decision Trees  using majority voting mechanism. They are also less prone to over-fitting/high-variance as each tree learns to predict differently.

# In[ ]:




