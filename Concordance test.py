
# coding: utf-8

# In[1]:

import re
import regex # Improved regular expression module that allows variable width lookahead and lookbehind
import numpy as np
import pandas as pd
pd.options.display.max_colwidth = 5000
pd.options.display.max_rows = 9999


# In[10]:

"""https://stackoverflow.com/questions/23555995/can-you-do-regex-with-concordance
   This allows to get concordance results using regex searches instead of NLTK's terrible concordance function"""
class RegExConcordanceIndex(object):
    "Class to mimic nltk's ConcordanceIndex.print_concordance."

    def __init__(self, text):
        self._text = text

    def print_concordance(self, reg, lines, width=80, demarcation=''):
        """
        Prints n <= @lines contexts for @regex with a context <= @width".
        Make @lines 0 to display all matches.
        Designate @demarcation to enclose matches in demarcating characters.
        """ 
        concordance = []
        matches = regex.finditer(reg, self._text, flags=re.M|re.I)
        
        if matches:
            for match in matches:
                start, end = match.start(), match.end()
                match_width = end - start
                remaining = (width - match_width) // 2
                if start - remaining > 0:
                    context_start = self._text[start - remaining:start]
                    #  cut the string short if it contains a newline character
                    context_start = context_start.split('\n')[-1]
                else:
                    context_start = self._text[0:start + 1].split('\n')[-1]
                context_end = self._text[end:end + remaining].split('\n')[0]
                concordance.append(context_start + demarcation + self._text
                                   [start:end] + demarcation + context_end)
                if lines and len(concordance) >= lines:
                    break
#             print("Displaying {} matches:".format(len(concordance)))
            return concordance
        else:
            return "No matches"


# In[3]:

ans = pd.DataFrame()
filename = r'biggest'
category = r'Social proof'
pat = r'big(er|est)?'
variations = r'big or biggest'
for l in range(1,5):
    df = pd.read_csv(r'C:\Users\Natarajan\Documents\Jupyter notebooks\Sears\Datasets\{}a.txt'.format(l),
                     header = None)
    df = df[df[1].str.contains(pat,regex = True,flags = re.I)]
    pat_col = []
    var_col = []
    ans = ans.append(df,ignore_index = True)
    for i in range(len(ans)):
        pat_col.append(pat)
        var_col.append(variations)
ans = ans.rename(index = str, columns = {0:'chatID',1:'concordance'})
ans['pattern'] = pat_col
ans['variations'] = var_col

ans
# ans.to_excel(r'C:\Users\Natarajan\Documents\Jupyter notebooks\Sears\Results\Results 061517\{}\{}.xlsx'.format(category,filename),
#              index = False)


# In[11]:

l = pd.Series()
b = pd.DataFrame()
filename = r'moderation'
pat = 'moderation'
for i in range(1,4):
    # Need to use utf-8 encoding or it won't work
    with open(r'C:\Users\Natarajan\Documents\Jupyter notebooks\Sears\Misc\tfidf\marinova{}.txt'.format(i),encoding = 'utf-8') as f:
        l = pd.Series(f.readlines())
    l = l.str.replace(r'\n','') # Replace all newlines with nothing
    # Here I tried to drop the empty rows but it didn't work
    l.replace(r'\r\n', np.nan, inplace=True)
    l.dropna(inplace = True)
    
    df = pd.DataFrame(l)
    df[0] = df[0].str.replace('\\t',' ') # Also didn't do much

    df = df[df[0].str.lower().str.contains(pat,regex = True,flags = re.I)]
    pat_col = []
    var_col = []
    b = b.append(df,ignore_index = True)
    name = []
    for j in range(len(b)):
        pat_col.append(pat)
        
b = b.rename(columns = {0:'concordance'})
b['pattern'] = pat_col
b.to_excel(r'C:\Users\Natarajan\Documents\Jupyter notebooks\Sears\Results\Results 063017\{}.xlsx'.format(filename),index = False)
b


# In[82]:

df = pd.read_table(r'C:\Users\Natarajan\Documents\Jupyter notebooks\Sears\Misc\Week 10\Jagdip_singh_raw_data2.txt',
                encoding = 'utf-8',header = None,sep = '$',names = ['Tag','Value'])
ab = df.loc[df['Tag'] == 'AB',:]
ab = ab.drop(ab[ab['Value'].str.contains('Copyright')].index)
concordance = []
pat = 'interaction'
filename = r'interaction'
for i in ab['Value']:
    concordance.append(list(RegExConcordanceIndex(i).print_concordance(pat,lines = 10,width = 200,demarcation = '**')))
concordance = list(filter(None, concordance))
res = pd.DataFrame(concordance)
res = (res[0].str.cat(sep = r'\n')+res[1].str.cat(sep = r'\n')).split(r'\n')
res = pd.DataFrame(res)
res = res.rename(columns = {0:'concordance'})
pat_list = [pat for i in range(len(res))]

res['pattern'] = pat_list
res.to_excel(r'C:\Users\Natarajan\Documents\Jupyter notebooks\Sears\Results\Results 071717\{}.xlsx'.format(filename),index = False)
res

