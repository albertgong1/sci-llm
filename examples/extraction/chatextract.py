'''
Example of ChatExtract implemenation, as described in the paper:
"Extracting Accurate Materials Data from Research Papers
with Conversational Language Models and Prompt Engineering"
by Maciej P. Polak and Dane Morgan
https://arxiv.org/abs/2303.05352

The code takes two arguments: the name of the csv file to analyze
and the name of the property to extract, for example:

python3 ./ChatExtract.py rc_data.csv "critical cooling rate"

the csv file requires to have at least 3 columns:
sentence,passage,doi

Where "sentence" is the sentence of the text that is to be analyzed
whether it does or does not contain data,
"passage" is a text passge composed of the papers title,
sentence previous to the one described above, and the sentence itself,
and "doi" which allows to identify the source of the text (just to keep track of data).

The output "extract_XX_XX_XX.csv" is a csv file containing extracted triplets of data
The output "results_XX_XX_XX.csv" file containing transcripts of all conversations
"binclas_XX_XX_XX.csv" contains the binary sentence classification.

a very short example rc_data.csv file with the corresponding output it also included.

'''


import pandas as pd
import sys
from re import split
from time import strftime, sleep
from copy import copy
import openai
dtime = strftime("%Y_%m_%d-%H%M%S")
openai.api_key = "YOUR_OPENAI_API_KEY"

START = 0

for i, arg in enumerate(sys.argv[1:]):
  if i==0:
    CSV_INPUT = sys.argv[i+1]
  if i==1:
    PROPERTY =  sys.argv[i+1]

dtime = dtime + "_" + CSV_INPUT

def prompt(Q,typ):
  if typ == 'yn':
    tkn = 6
  elif typ == 'all':
    tkn = 500
  elif typ == 'tab':
    tkn = 500
  while True:
    try:
      response = openai.ChatCompletion.create(
        model="gpt-4-0314",
        messages=Q,
        temperature=0,
        max_tokens=tkn,
        frequency_penalty=0,
        presence_penalty=0
      )
      break
    except Exception as e:
      print("An error occurred:", e)
      if 'Please reduce the length of the messages' in str(e):
        print('TRUNCATING')
        if 'Use only data present in the text. If data is not present in the text, type' in Q[1]["content"]:
          print(Q.pop(3))
          print(Q.pop(3))
        else:
          print(Q.pop(1))
          print(Q.pop(1))
      elif 'per min' in str(e):
          print("Sleeping for 15 sec.")
          sleep(15)
  return(Q,response['choices'][0]['message']['content'])

test_df = pd.read_csv(CSV_INPUT)

classif_q = 'Answer "Yes" or "No" only. Does the following text contain a value of '+PROPERTY+'?\n\n'
ifmulti_q = 'Answer "Yes" or "No" only. Does the following text contain more than one value of '+PROPERTY+'?\n\n'
single_q = [
'Give the number only without units, do not use a full sentence. If the value is not present in the text, type "None". What is the value of the '+PROPERTY+' in the following text?\n\n',
'Give the unit only, do not use a full sentence. If the unit is not present in the text, type "None". What is the unit of the '+PROPERTY+' in the following text?\n\n',
'Give the name of the material only, do not use a full sentence. If the name of the material is not present in the text, type "None". What is the material for which the '+PROPERTY+' is given in the following text?\n\n'
]
singlefollowup_q = [
['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ',' the value of the '+PROPERTY+' for the compound in the following text?\n\n'],
['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ',' the unit of the value of '+PROPERTY+' in the following text?\n\n'],
['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is "','" the compound for which the value of '+PROPERTY+' is given in the following text? Make sure it is a real compound.\n\n']
]

tab_q = 'Use only data present in the text. If data is not present in the text, type "None". Summarize the values of '+PROPERTY+' in the following text in a form of a table consisting of: Material, Value, Unit\n\n'
tabfollowup_q = [
['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is "','" the ',' compound for which the value of '+PROPERTY+' is given in the following text? Make sure it is a real compound.\n\n'],
['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ',' the value of the '+PROPERTY+' for the ',' compound in the following text?\n\n'],
['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ',' the unit of the ',' value of '+PROPERTY+' in the following text?\n\n']
]

unifymat_q = 'From the following phrase extract a uniquely identifiable material composition only, do not provide any other detals than the material composition. If a unique material composition is not possible to extract, type "None". Phrase: ' 
goodmat_q = [
['Answer "Yes" or "No" only. Be very strict. Is ',' a uniquely identifiable material?\n\n'],
['Answer "Yes" or "No" only. Be very strict. Is ',' a specific material?\n\n'],
['Answer "Yes" or "No" only. Be very strict. Is ',' a precisely specified material?\n\n'],
['Answer "Yes" or "No" only. Be very strict. Is ',' an unabiguous material?\n\n'],
['Answer "Yes" or "No" only. Be very strict. Is ',' a well-defined material?\n\n']
]
goodmat_q=[]
it = [ 'first','second','third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth']
col=['Material','Value','Unit']

single_cols = ['value','unit','material']

ntot=len(test_df)
for i in range(START,len(test_df)):
  try:
    binary_classif=[]
    answers = []
    sss=[{"role": "system", "content": ""}]
    print("Processing ",CSV_INPUT," ",i," ",round(i/ntot*100,1),"%")
    ss = classif_q+test_df["sentence"][i]
    sss.append({"role": "user", "content": ss})
    sss,ans = prompt(sss,'yn')
    sss.append({"role": "assistant", "content": ans})
    if 'yes' in ans.strip().lower():
      binary_classif.append(1)
      result = {}
      ss = ifmulti_q+test_df["passage"][i]
      sss.append({"role": "user", "content": ss})
      sss,ans = prompt(sss,'yn')
      sss.append({"role": "assistant", "content": ans})
      if 'no' in ans.lower():
        result["passage"] = [test_df["passage"][i]]
        result["doi"] = [test_df["doi"][i]]
        result["material"] =[]
        result["value"] =[]
        result["unit"] =[]
        result["material_valid"] =[]
        result["value_valid"] =[]
        result["unit_valid"] =[]
        for j in range(len(single_q)):
          ss = single_q[j]+test_df["passage"][i]
          sss.append({"role": "user", "content": ss})
          sss,ans = prompt(sss,'all')
          sss.append({"role": "assistant", "content": ans})
          result[single_cols[j]].append(ans)
          if 'none' in ans.lower():
            result[single_cols[j]+"_valid"].append(0)
          else:
            result[single_cols[j]+"_valid"].append(1)
      elif 'yes' in ans.lower():
        ss = tab_q+test_df["passage"][i]
        sss.append({"role": "user", "content": ss})
        sss,tab = prompt(sss,'tab')
        sss.append({"role": "assistant", "content": tab})
        sst = copy(sss)
        tab = [split('[,|]',row) for row in tab.strip().split('\n')]
        tab = [[item.strip() for item in row if len(item.strip())>0] for row in tab if len(row)>=3]
        if len(tab)<=0:
          tab.append(['Material','Value','Unit'])
        if len(tab)<=1:
          tab.append(['None','None','None'])
        else:
          tab.pop(1)
        head = tab.pop(0)
        tab = pd.DataFrame(tab,columns=head)
        result["passage"] = []
        result["doi"] = []
        result["material"] = []
        result["value"] = []
        result["unit"] = []
        result["material_valid"] = []
        result["value_valid"] = []
        result["unit_valid"] = []
        for k in range(len(tab)):
          sst.append({"role": "tab", "content": tab[col[0]][k]+","+tab[col[1]][k]+","+tab[col[2]][k]})
          result["passage"].append(test_df["passage"][i])
          result["doi"].append(test_df["doi"][i])
          multi_valid = True
          for l in range(3):
            ss = tabfollowup_q[l][0]+str(tab[col[l]][k])+tabfollowup_q[l][1]+it[k]+tabfollowup_q[l][2]+test_df["passage"][i]
            result[col[l].lower()].append(tab[col[l]][k])
            if 'none' in tab[col[l]][k].lower():
              result[col[l].lower()+"_valid"].append(0)
              multi_valid = False
            elif multi_valid:
              sss.append({"role": "user", "content": ss})
              sst.append({"role": "user", "content": ss})
              sss,ans = prompt(sss,'yn')
              sss.append({"role": "assistant", "content": ans})
              sst.append({"role": "assistant", "content": ans})
              if 'no' in ans.lower():
                result[col[l].lower()+"_valid"].append(0)
                multi_valid = False
              else:
                result[col[l].lower()+"_valid"].append(1)
            else:
              result[col[l].lower()+"_valid"].append(1)
      try:
        if i==0:
          pd.DataFrame(result).to_csv("extracted_"+dtime, mode='a', index=False, header=True)
        else:
          pd.DataFrame(result).to_csv("extracted_"+dtime, mode='a', index=False, header=False)
      except Exception as e:
        print('Appending extracted gone wrong: ',i,"  ",e)
        print('Appending extracted gone wrong: ',result,"  ",e)
        print('Appending extracted gone wrong: ',tab,"  ",e)
    else:
      binary_classif.append(0)
    pd.DataFrame(binary_classif).to_csv("binclas_"+dtime, mode='a', index=False, header=False)
    try:
      pd.DataFrame(sst).to_csv("results_"+dtime, mode='a', index=False, header=False)
      del sst
    except:
      pd.DataFrame(sss).to_csv("results_"+dtime, mode='a', index=False, header=False)
  except Exception as e:
    print("GENERAL ERROR, ignoring and proceeding to next line",e)
