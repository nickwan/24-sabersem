from IPython.display import display, HTML, Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pybaseball as bb

import catboost as cb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, percentileofscore
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm

from scipy.stats import pearsonr
import itertools

def increase_font():
  from IPython.display import Javascript
  display(Javascript('''
  for (rule of document.styleSheets[0].cssRules){
    if (rule.selectorText=='body') {
      rule.style.fontSize = '24px'
      break
    }
  }
  '''))
get_ipython().events.register('pre_run_cell', increase_font)

def stuff_features(project_dir):
  display(Image(filename=f'{project_dir}/stuffs.png'))   

  urls = """
  https://github.com/saberpowers/talks/blob/master/2023/saberseminar/slides.pdf

  https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/

  https://www.nytimes.com/athletic/2641834/2021/06/11/the-pitcher-report-what-exactly-is-stuff-featuring-rich-hill-sam-long-and-more/
  """

  print(urls)

def draw_sz(sz_top=3.5, sz_bot=1.5, ls='k-'):
  """
  draw strike zone
  draw the strike zone on a plot using mpl
  inputs:
    sz_top: top of strike zone (ft)
    sz_bot: bottom of strike zone (ft)
    ls: linestyle (use `plt.plot()` linestyle conventions)
  output:
    strike zone plot
  """
  plt.plot([-0.708, 0.708], [sz_bot,sz_bot], ls)
  plt.plot([-0.708, -0.708], [sz_bot,sz_top], ls)
  plt.plot([0.708, 0.708], [sz_bot,sz_top], ls)
  plt.plot([-0.708, 0.708], [sz_top,sz_top], ls)

def draw_home_plate(catcher_perspective=True, ls='k-'):
  """
  draw home plate from either the catcher perspective or pitcher perspective
  inputs:
    catcher_perspective: orient home plate in the catcher POV. if False, orients
      home plate in the pitcher POV.
    ls: linestyle (use `plt.plot()` linestyle conventions)
  output:
    home plate plot
  """
  if catcher_perspective:
    plt.plot([-0.708, 0.708], [0,0], ls)
    plt.plot([-0.708, -0.708], [0,-0.3], ls)
    plt.plot([0.708, 0.708], [0,-0.3], ls)
    plt.plot([-0.708, 0], [-0.3, -0.6], ls)
    plt.plot([0.708, 0], [-0.3, -0.6], ls)
  else:
    plt.plot([-0.708, 0.708], [0,0], ls)
    plt.plot([-0.708, -0.708], [0,0.1], ls)
    plt.plot([0.708, 0.708], [0,0.1], ls)
    plt.plot([-0.708, 0], [0.1, 0.3], ls)
    plt.plot([0.708, 0], [0.1, 0.3], ls)


def get_fg_data():
  fg = bb.fg_pitching_data(start_season=2024, qual=10)
  ids = bb.playerid_reverse_lookup(fg['IDfg'].unique(),key_type='fangraphs')
  ids = ids.loc[:, ['key_mlbam', 'key_fangraphs']].rename(columns={'key_mlbam':'pitcher', 'key_fangraphs':'IDfg'})
  fg = ids.merge(fg)
  return fg

def do_corr(coly,lb2,stuff):  
  fig,ax = plt.subplots(1,2,sharex=False, sharey=True, figsize=(15, 5))
  r2 = pearsonr(lb2[stuff],lb2[coly])[0]**2
  sns.regplot(data=lb2, x=stuff, y=coly, scatter=False,line_kws={'color':'k'},ax=ax[0])
  sns.scatterplot(data=lb2, x=stuff, y=coly, ax=ax[0], color=(4/255, 118/255, 29/255))
  ax[0].set_title(f"R2: {round(r2,3)}")
  sns.despine(ax=ax[0])

  r2 = pearsonr(lb2['Stuff+'],lb2[coly])[0]**2
  sns.regplot(data=lb2, x='Stuff+', y=coly, scatter=False,line_kws={'color':'k'},ax=ax[1])
  sns.scatterplot(data=lb2, x='Stuff+', y=coly, ax=ax[1], color=(143/255, 133/255, 255/255))
  ax[1].set_title(f"R2: {round(r2,3)}")
  sns.despine(ax=ax[1])

  plt.show()

def youtube(project_dir):
  display(Image(filename=f'{project_dir}/youtube.png'))   

  urls = """
  https://www.youtube.com/watch?v=pocp0KdrIdk&list=PL6PX3YIZuHhwo48MyTASIor4j5NV7qq1W&index=3

  """
  print(urls)  
  display(Image(filename=f'{project_dir}/youtube-qr.png'))   

def kaggle(project_dir):
  display(Image(filename=f'{project_dir}/kaggle1.png'))   
  display(Image(filename=f'{project_dir}/kaggle2.png'))   

  urls = """
  https://www.kaggle.com/competitions/nwds-batted-balls/overview

  """
  print(urls)  


def clean_data(df):

  text = """
  The data cleaning I do:  
  - Left-handed pitching considerations: flip values for pfx_x, release_pos_x 
  
  - More LHP considerations: spin_axis needs to be flipped, but it's in degrees 
    so be mindful! Multiply by -1, add 360, replace 360 with 0 
  
  - To make differences from "primary pitch" I just use FB
    Filter for FF and SI, group by pitcher, get averages for velo and break
    Merge back onto your original dataframe
    Subtract the averages from each pitch 

  - Code is available on GitHub, there's also several VODs on Twitch

  """
  print(text)

  for col in ['pfx_x','release_pos_x']:
    df[f'{col}_adj'] = df[col].copy()
    df.loc[df['p_throws']==1, f'{col}_adj'] = df.loc[df['p_throws']==1, f'{col}_adj'].mul(-1)

  df['spin_axis_adj'] = df['spin_axis'].copy()
  df.loc[df['p_throws']==1, f'spin_axis_adj'] = df.loc[df['p_throws']==1, f'spin_axis_adj'].mul(-1).add(360)
  df.loc[:,'spin_axis_adj'] = df.loc[:,'spin_axis_adj'].replace(360, 0)

  _df = df.loc[df['pitch_type'].isin(['FF','SI']), ['pitcher','release_speed','pfx_x_adj','pfx_z']].groupby(['pitcher'],as_index=False).mean().rename(columns={'release_speed':'release_speed_avg','pfx_x_adj':'pfx_x_adj_avg','pfx_z':'pfx_z_avg'})
  df = df.merge(_df)

  for col in ['release_speed','pfx_x_adj', 'pfx_z']:
    df[f'{col}_diff'] = df[col].sub(df[f'{col}_avg'])

  return df

def get_model_data(df):
  text = """
  Features going into the model:

    'release_speed_diff', 'pfx_x_adj_diff', 'pfx_z_diff',
    'release_speed', 'pfx_x', 'pfx_z','release_spin_rate',
    'spin_axis_adj', 'release_pos_x_adj','release_extension',
    'release_pos_z'

  Model target (DV):
    'delta_run_exp'

  Model parameters:  
    border_count=100 
    depth=8 
    reg_lambda=6 
    eta=0.008 
    n_estimators=300 
    loss_function='MultiRMSE' 

  """

  print(text)

  feats = [
      'release_speed_diff', 'pfx_x_adj_diff', 'pfx_z_diff',
      'release_speed', 'pfx_x', 'pfx_z','release_spin_rate',
      'spin_axis_adj', 'release_pos_x_adj','release_extension',
      'release_pos_z'
  ]

  target = 'delta_run_exp'
  model_data = (df.loc[(df['on_3b'].isna()) & (df['on_2b'].isna()) & (df['on_1b'].isna()) & (df['release_speed']>65) & (df['plate_x'].between(-1.5,1.5)) & (df['plate_z'].between(-1,6))].dropna(subset=feats+[target]))
  params = dict(
      verbose=False, 
      border_count=100, 
      depth=8, 
      l2_leaf_reg=6, 
      learning_rate=0.008, 
      n_estimators=300, 
      loss_function='MultiRMSE'
  )
  return model_data, feats, target, params 

def make_stuff(df, stuff, target):
  text = '''
  Once you have an output, you can convert that in a variety of ways

  "Plus" 
  Take the target average of all pitches 
  Divide pitches by avg 

  "Delta" 
  Take the target average of each pitch type for a pitcher 
  Subtract pitches by avg 

  "$$$"
  Use non-rookie salaries to calculate dollars per run 
  Since the target is pitch-level runs, multiply by dollars per run 

  "ERA-ify"
  Take the target average of all pitches 
  Divide your favorite number by the target average 
  Multiply pitches by newly developed constant 

  Unsolicited advice:
  Converting out of your target should have a purpose 
  Conversions can introduce new artifacts 
  Be mindful of your "why" 

  '''
  print(text)
  df['_stage'] = df[f'{target}_cb'].mul(-1)
  df['_stage'] = df['_stage'].add(df['_stage'].abs().max())
  df[stuff] = (df['_stage'].div(df['_stage'].mean()))*100
  return df

 
def make_leaderboard(df,target,stuff):
  lb = df.loc[:, ['pitcher','player_name', 'pitch_name', target,stuff]].groupby(['pitcher','player_name', 'pitch_name'], as_index=False).mean()
  lb = lb.sort_values(stuff,ascending=False).reset_index(drop=True)
  return lb 

def is_it_good(df,target,stuff,project_dir):
  # this line of code 
  # fg = get_fg_data() 
  # is how i would access the fangraphs data but it takes about a minute and i didn't want to incur a time loss
  fg = pd.read_csv(f'{project_dir}/fg.csv')
  lb2 = df.loc[:, ['pitcher','player_name', target,stuff]].groupby(['pitcher','player_name'], as_index=False).mean()
  lb2 = lb2.merge(fg.loc[:, ['pitcher', 'Team','K%','BB%', 'ERA', 'FIP', 'xFIP', 'Stuff+', 'WAR', 'Dollars']])
  lb2['Dollars'] = lb2['Dollars'].str.replace('$', '').str.replace('(', '').str.replace(')', '').astype(float)
  colsy = [
      'K%', 'BB%', 'ERA', 'FIP', 'xFIP', 'WAR', 'Dollars', stuff
  ]
  for y in colsy:
    do_corr(y,lb2,stuff)
  return lb2 
