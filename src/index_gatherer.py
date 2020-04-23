
# IMPORT PACKAGES 
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() 
from functools import partial, reduce 
from abc import ABC, abstractmethod
 
# DEFINE ABSTRACT CLASS
class gatherer(ABC):
  @abstractmethod
  def get_tickers_info(self):
    pass

  @abstractmethod
  def query_tickers(self):
    pass

	# DEFINE BASE GATHERER
class index_gatherer(gatherer):
  """
  This is the abstract class for our connectors. 
  They must be created with at least one methods:

  query_tickers: data requesting from Yahoo! Finance or any other source.
  """
  def __init__(self, components_url):
    self.components_url = components_url 
    self.index_info_table = None
    self.components_list = None
    self.ticker_data = None
    super().__init__()
    
  def get_tickers_info(self, tbl_index = 0, ticker_column = 'Symbol'):
    """
    This method requests the components of the index from a url in the form of
    HTML tables and extracts the desired index. 

    It stores in the components_list attribute a python list with all the symbols
    that are the current constituents of the index 
    """
    tables = pd.read_html(self.components_url)
    self.index_info_table = tables[tbl_index]
    self.components_list = self.index_info_table[ticker_column].values

  def query_tickers(self, output_format, data_src):
    """
    This method is extended to request the tickers from Yahoo! Finance. 
    Other sources could also be easily integrated

    Parameters
    ----------
    output_format: `str` with either 'long'(Rc) or 'wide'(rC) output format 
    data_src: `str` source to query from. One of ['yahoo','quandl']
    """ 
    assert data_src in ['yahoo', 'quandl']

    if data_src == 'yahoo':
      self.ticker_data = pdr.get_data_yahoo(list(self.components_list))
    
    if output_format == 'long':
      self.ticker_data = self.ticker_data.reset_index().melt(id_vars = 'Date').dropna().rename(columns = {'variable_1':'Symbol'})
      self.ticker_data = self.ticker_data.pivot_table(index = ['Date', 'Symbol'], columns = 'variable_0')
      self.ticker_data.columns = [i[1] for i in self.ticker_data.columns]


import numpy as np 
def transformer_pct(df):
  """
  This function transforms all numeric columns to intraday %change
   
  parameters
  ----------
  df: `pandas.DataFrame`
  """
  for i in df.columns:
    if np.issubdtype(df[i].dtype, np.number):
      df[i] = df[i].pct_change()
  return df

def compute_weights_ew(df, grouping_vars):
  """
  This function creates the weights to compute a simple average
   
  parameters
  ----------
  df: `pandas.DataFrame`
  """  
  df['weight'] = 1
  df = df.loc[:,grouping_vars+['weight']].groupby(grouping_vars).sum().reset_index().merge(df, on = grouping_vars, suffixes = ['_sum','_ind'])
  df['weight']=df['weight_ind']/df['weight_sum']
  df = df.loc[:,grouping_vars + ['weight', 'Symbol']]
  return df

def aggregate_ew(df, weights, grouping_vars):
  # Set all non-numeric columns as index
  cols_df = [i for i in df.columns.tolist() if not np.issubdtype(df[i].dtype, np.number)]
  cols_wgt = [i for i in weights.columns.tolist() if not np.issubdtype(weights[i].dtype, np.number)]
  common = list(set(cols_df).intersection(set(cols_wgt)))

  df = df.set_index(common)
  weights = weights.set_index(common)

  duplicates = list(set(df.columns).intersection(set(weights.columns)))
  if len(duplicates)>0:
    df = df.drop(duplicates, axis = 1)

  # merge df info and weights
  df = df.merge(weights, right_index = True, left_index = True)
  
  scale = df['weight']

  # apply aggregation
  df = df.apply(lambda x: x*scale)
  cols= df.columns.tolist()
  df = df.reset_index().loc[:,cols+grouping_vars].groupby(grouping_vars).sum()
  return df

class SP500_gatherer(index_gatherer):
  """
  This class is used as a connector to obtain data for the SP500
  """
  def __init__(self, components_url):
        index_gatherer.__init__(self, components_url)
        self.sector_information = None

  def ew_pipeline(self, date_var, grouping_vars):
  """
  This method is used to compute the daily average returns for every GICS Sector 
  """
    df_data = self.ticker_data.sort_index(level =[1,0]).reset_index().merge(self.index_info_table.loc[:,grouping_vars+['Symbol']], on = 'Symbol', how = 'left')
    wgts = compute_weights_ew(df_data, grouping_vars+['Date'])
    self.sector_information = aggregate_ew(transformer_pct(df_data), wgts, grouping_vars+[date_var])

