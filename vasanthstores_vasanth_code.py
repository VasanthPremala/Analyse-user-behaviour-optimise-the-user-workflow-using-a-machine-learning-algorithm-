#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


df=pd.read_csv("vasanthstores_vasanth_data.csv")


# In[15]:


df.head()


# In[16]:


df.set_index(df.columns[0],inplace=True)


# In[17]:


df.head()


# In[18]:


data=[]
for i in range(100):
    x=df.values[i]
    x=x[~pd.isnull(x)]
    data.append(x)
data


# In[19]:


from markovclick.models import MarkovClickstream
m = MarkovClickstream(data)


# In[20]:


fig=plt.figure(figsize=(18, 16), dpi= 100, facecolor='r', edgecolor='k')
sns.heatmap(m.prob_matrix, xticklabels=m.pages, yticklabels=m.pages,cmap="YlGnBu")


# In[21]:


from markovclick.viz import visualise_markov_chain
graph = visualise_markov_chain(m)


# In[22]:


print(graph.source)


# In[23]:


graph.render('test-output/round-table.gv', view=True)
graph


# In[ ]:




