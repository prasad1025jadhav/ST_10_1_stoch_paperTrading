
import yfinance as yf
import ccxt
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import pandas_ta as ta
import matplotlib.pyplot as plt
import math
from stocktrends import Renko
import requests
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')

#### DHAN Data
from dhanhq import dhanhq
import websocket
import mibian


dhan = dhanhq("1101927963","eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzUyNzcyMTYyLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMTkyNzk2MyJ9.HUvWwY8iUALo6HjsK1DhR4OUUvXxx7HEpiuipHhH1k_iyGlbQzaw2p_7xz0lMsq0EDP8g0RVe7p9Tvj6ewtT2A")


### Pre-requiste data:= = = = = == = = = = = = = = = = = =(Live)

######  data 1: Expiry Date list NIFTY:
response_nifty_exp_list= dhan.expiry_list(
    under_security_id=13,                       # Nifty
    under_exchange_segment="IDX_I"
)
nifty_expiry_list = pd.DataFrame(response_nifty_exp_list['data']['data'], columns=['expiry_dates'])

def get_next_thursday(from_date=None):
    if from_date is None:
        from_date = datetime.today()
    else:
        from_date = datetime.strptime(from_date, "%Y-%m-%d")

    days_ahead = (3 - from_date.weekday() + 7) % 7  # Thursday is weekday 3
    days_ahead = days_ahead or 7  # If already Thursday, move to next Thursday
    next_thursday = from_date + timedelta(days=days_ahead)
    next_thursday= next_thursday.date().strftime('%Y-%m-%d') 
    return next_thursday

# Example usage:
#print(get_next_thursday("2025-06-11"))  # Output: 2025-06-19



    
#####  data 2: Security list data:= = = = = == = = = = = = = = = = = =
securities_data=dhan.fetch_security_list("compact")   # RAW NSE LIST (LIVE)

#securities_data=pd.read_csv("/Users/prasad.jadhav/Desktop/Prasad/Trading/New_start/Security_list_jun25.csv")
securities_data=securities_data[ (securities_data['SEM_EXM_EXCH_ID']=='NSE') & 
                                (securities_data['SEM_INSTRUMENT_NAME']=='OPTIDX') &
                                (securities_data['SEM_TRADING_SYMBOL'].str.startswith('NIFTY')) ] 

securities_data['SEM_EXPIRY_DATE']= pd.to_datetime(securities_data['SEM_EXPIRY_DATE'], errors='coerce').dt.strftime('%Y-%m-%d')



### Pre-requiste assumpations:= = = = = == = = = = = = = = = = = =
month_ago=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')    
week_ago=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')    
yesterday=(datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')    
today=datetime.now().strftime('%Y-%m-%d')    

# Intraday Minute Data
security_id=13  # NIFTY INDEX Data
exchange_segment='IDX_I'
instrument_type='INDEX'
from_date=month_ago #from which u want analysis
to_date=today


response=dhan.intraday_minute_data(security_id,exchange_segment,instrument_type, from_date,to_date)

Nifty_index_data = pd.DataFrame(response['data'])
Nifty_index_data['timestamp'] = pd.to_datetime(Nifty_index_data['timestamp'], unit='s',utc=True)
Nifty_index_data['timestamp'] = Nifty_index_data['timestamp'].dt.tz_convert('Asia/Kolkata')
Nifty_index_data['timestamp'] = Nifty_index_data['timestamp'].dt.tz_localize(None) # remove +5:30 from date time
Nifty_index_data=Nifty_index_data.round()

train_data=Nifty_index_data.iloc[1:round(len(Nifty_index_data)*0.5)]
test_data=Nifty_index_data.iloc[ (round(len(Nifty_index_data)*0.5)+1) : len(Nifty_index_data)]


#1 apply indicator on raw data:- - - - - - - - - - - - - -- - - - - - - -- - - - - - - - - - - - - -- - - - - - - -
def apply_indicators(data):
    # Indicators    
    data['EMA_9'] =round(ta.ema(data['low'],length=9))
    data['EMA_20'] =round(ta.ema(data['low'],length=20))
   # data['EMA_20_high'] = round(ta.ema(data['high'],length=20))
    data['EMA_50'] = round(ta.ema(data['high'],length=50))
    stoch_rsi=ta.stochrsi(close=data['close'], length=14, rsi_length=14, k=3, d=3)
    stoch_rsi.columns = ['stochrsi_k', 'stochrsi_d']
    data = pd.concat([data, stoch_rsi], axis=1)
    #data['EMA_200'] = data.ta.ema(length=ema_200)
    data['RSI'] = data.ta.rsi(length=14)
    supertrend_10_2 = round(data.ta.supertrend(length=10, multiplier=2))
    data['supertrend_10_2'] = supertrend_10_2.iloc[:,0]
    supertrend_10_1 = round(data.ta.supertrend(length=10, multiplier=1))
    data['supertrend_10_1'] = supertrend_10_1.iloc[:,0]
    supertrend_10_3 = round(data.ta.supertrend(length=10, multiplier=3))
    data['supertrend_10_3'] = supertrend_10_3.iloc[:,0]
    supertrend_40_10 = round(data.ta.supertrend(length=40, multiplier=10))
    data['supertrend_40_10'] = supertrend_40_10.iloc[:,0]
    supertrend = ta.supertrend(high=data['high'], low=data['low'], close=data['close'], length=40, multiplier=10)
    supertrend.columns=['supertrend', 'upper_band', 'lower_band', 'trend_direction']
    data = data.join(supertrend)

    return data
    

##2: calculate ADX: - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - -- - - - - - - -
def calculate_adx(df, period=14):
    # Calculate True Range (TR)
    df["high-low"] = df["high"] - df["low"]
    df["high-close"] = abs(df["high"] - df["close"].shift(1))
    df["low-close"] = abs(df["low"] - df["close"].shift(1))
    df["TR"] = df[["high-low", "high-close", "low-close"]].max(axis=1)
    
    # Calculate +DM and -DM
    df["+DM"] = np.where((df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]),
                         np.maximum(df["high"] - df["high"].shift(1), 0), 0)
    df["-DM"] = np.where((df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)),
                         np.maximum(df["low"].shift(1) - df["low"], 0), 0)
    
    # Smooth TR, +DM, and -DM using an EMA-like formula
    df["TR_Smoothed"] = df["TR"].rolling(window=period).mean()
    df["+DM_Smoothed"] = df["+DM"].rolling(window=period).mean()
    df["-DM_Smoothed"] = df["-DM"].rolling(window=period).mean()
    
    # Calculate +DI and -DI
    df["+DI"] = (df["+DM_Smoothed"] / df["TR_Smoothed"]) * 100
    df["-DI"] = (df["-DM_Smoothed"] / df["TR_Smoothed"]) * 100
    
    # Calculate DX
    df["DX"] = (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])) * 100
    
    # Calculate ADX
    df["ADX"] = df["DX"].rolling(window=period).mean()
    
    # Drop intermediate columns for clarity (optional)
    df.drop(["high-low", "high-close", "low-close", "+DM", "-DM", "TR_Smoothed", "+DM_Smoothed", "-DM_Smoothed", "DX"], axis=1, inplace=True)
    
    return df
   


#3 Create RENKO chart data:- - - - - - - - - - - - - -- - - - - - - -- - - - - - - - - - - - - -- - - - - - - -
from stocktrends import Renko
from scipy.signal import argrelextrema
 
# Brick size for calculations:
brick_size = 10

def get_renko(df, brick_size):
    df_reset = df.reset_index()
    renko_df = df_reset[['timestamp', 'open', 'high', 'low', 'close']]
    renko_df.columns = ['date', 'open', 'high', 'low', 'close']
    renko = Renko(renko_df)
    renko.brick_size = brick_size
    renko.chart_type = Renko.PERIOD_CLOSE
    renko_ohlc=renko.get_ohlc_data()
    renko_ohlc['trend']= np.where(renko_ohlc['uptrend'],"UP","DOWN")
    renko_ohlc=apply_indicators(renko_ohlc)  
    renko_ohlc=calculate_adx(renko_ohlc)
    
    ### Resistance and support level:
    renko_ohlc['resistance_level'] = np.nan
    renko_ohlc['support_level'] = np.nan

    # Use order=2 or 3 depending on sensitivity
    highs = argrelextrema(renko_ohlc['close'].values, np.greater_equal, order=2)[0]
    lows = argrelextrema(renko_ohlc['close'].values, np.less_equal, order=2)[0]

    # Assign swing highs/lows as support/resistance
    renko_ohlc.loc[renko_ohlc.index[highs], 'resistance_level'] = renko_ohlc.iloc[highs]['close']
    renko_ohlc.loc[renko_ohlc.index[lows], 'support_level'] = renko_ohlc.iloc[lows]['close']
    
    renko_ohlc['support_level_ffill'] = renko_ohlc['support_level'].ffill()
    renko_ohlc['resistance_level_ffill'] = renko_ohlc['resistance_level'].ffill()
    
    
    ## calculate variables:
    renko_ohlc['diff_cp_9ema']= round(renko_ohlc['close'] - renko_ohlc['EMA_9'],0)   
    renko_ohlc['diff_cp_20ema']= round(renko_ohlc['close'] - renko_ohlc['EMA_20'],0)
    renko_ohlc['diff_cp_50ema']= round(renko_ohlc['close'] - renko_ohlc['EMA_50'],0)
    renko_ohlc['diff_20_50ema']= round(renko_ohlc['EMA_20'] - renko_ohlc['EMA_50'],0)
    renko_ohlc['diff_9_20ema']= round(renko_ohlc['EMA_9'] - renko_ohlc['EMA_20'],0)
    renko_ohlc['supertrend_10_2']= round(renko_ohlc['supertrend_10_2'],0)
    renko_ohlc['supertrend_10_3']= round(renko_ohlc['supertrend_10_3'],0)
    renko_ohlc['supertrend_40_10']= round(renko_ohlc['supertrend_40_10'],0)
    renko_ohlc['RSI']= round(renko_ohlc['RSI'] ,0)
    renko_ohlc['diff_cp_supertrend_10_2']= round(renko_ohlc['close'] - renko_ohlc['supertrend_10_2'],0) 
    renko_ohlc['diff_cp_supertrend_10_3']= round(renko_ohlc['close'] - renko_ohlc['supertrend_10_3'],0) 
    renko_ohlc['diff_cp_supertrend_40_10']= round(renko_ohlc['close'] - renko_ohlc['supertrend_40_10'],0) 
    renko_ohlc['diff_cp_st_10_1']= round(renko_ohlc['close'] - renko_ohlc['supertrend_10_1'],0) 
    renko_ohlc['diff_stoch_f_d']= round(renko_ohlc['stochrsi_k'] - renko_ohlc['stochrsi_d'],0) 
    renko_ohlc['diff_di']= round(renko_ohlc['+DI'] - renko_ohlc['-DI'],0) 
    
    # DELTA Variables:
    renko_ohlc['DI_delta']= round(renko_ohlc['+DI'] - renko_ohlc['+DI'].shift(1),0)
    renko_ohlc['ema9_delta']= round(renko_ohlc['EMA_9'] - renko_ohlc['EMA_9'].shift(1),0)
    
    return renko_ohlc

#4 #Find HH and HL : - - - - - - - - - - - - - -- - - - - - - -- - - - - - - - - - - - - -- - - - - - - -
def find_high_low_by_sequence(renko_data):
    """
    Groups the data by sequences of "up" and "down" bricks, 
    and finds the high/low of the first brick in "up" sequences
    and the high/low of the last brick in "down" sequences.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['type', 'high', 'low'].

    Returns:
        pd.DataFrame: A summary DataFrame with high/low of first "up" brick and last "down" brick in each sequence.
    """
    # Assign a group ID to consecutive "up" and "down" bricks
    renko_data['group_id'] = (renko_data['trend'] != renko_data['trend'].shift()).cumsum()

    # Create empty lists to store results
    up_summary = []
    down_summary = []

    # Group by group_id
    grouped = renko_data.groupby('group_id')
    g_data=renko_data.groupby('group_id').size().reset_index()
    last_group_id= g_data.iloc[-1]['group_id']
    
    for group_id, group in grouped:
        first_row = group.iloc[0]
        last_row = group.iloc[-1]

        # If the group is an "up" sequence
        if first_row['trend'] == 'UP':
            up_summary.append({
                'group_id': group_id,
                'trend': 'UP',
                'hh': last_row['close'],
                'RSI':last_row['RSI']
            })

        # If the group is a "down" sequence
        if first_row['trend'] == 'DOWN':
            down_summary.append({
                'group_id': group_id,
                'trend': 'DOWN',
                'hl': last_row['close'],              
                'RSI':last_row['RSI']
            })    

    # Create DataFrames for "up" and "down" summaries
    up_df = pd.DataFrame(up_summary)
    down_df = pd.DataFrame(down_summary)

    # Merge the two summaries
    result_df = pd.concat([up_df, down_df], ignore_index=True).sort_values('group_id')
    result_df=result_df[result_df['group_id']!=last_group_id]
    result_df=result_df.drop_duplicates(['trend','group_id'], keep='last') 
    return result_df


## Buying signal: - - - - - - - - - - - - - -- - - - - - - -- - - - - - - - - - - - - -- - - - - - - -    
def check_buy_conditions_kiran(renko_data):
  
    current_row = pd.DataFrame([renko_data.iloc[-1]])
    current_price = int(current_row['close'].iloc[0])  # or .item()/.values[0]

    prev_row = pd.DataFrame([renko_data.iloc[-2]])
    prev_price = int(prev_row['close'].iloc[0])
    
    # Convert conditions to scalar booleans
    bull_cond = ( 
      ( current_row['diff_stoch_f_d'].iloc[0] >=0) &
       (current_row['diff_cp_st_10_1'].iloc[0] >0) &
       (current_row['diff_cp_9ema'].iloc[0] >= 5) &
       (current_row['diff_9_20ema'].iloc[0] > 0) &
       (current_row['ADX'].iloc[0] >= 20)
      )
    
    bear_cond = (
        (current_row['diff_stoch_f_d'].iloc[0] < 0) &
        (current_row['diff_cp_st_10_1'].iloc[0] <0) &
        (((current_row['+DI'].iloc[0] - current_row['-DI'].iloc[0])<0 ) | (current_row['DI_delta'].iloc[0] < 0)) &
        (current_row['ADX'].iloc[0] >= 20) &
        (current_row['diff_cp_9ema'].iloc[0] <= -5) &
        (current_row['diff_9_20ema'].iloc[0] < -1) 
        )

    # Simplified conditional logic
    action = "Bull" if bull_cond else "Bear" if bear_cond else "no"
    
    return action # Always re    


### # Expiry List of Underlying ---------------------------------- - - - - -- -- - - --- ------ -- - - - 

def option_strike_price_selection_Live(ltp, option_type,trade_date, step=50,strikes=3, symbol='NIFTY' ):

    
   ## current_expiry=nifty_expiry_list['expiry_dates'].iloc[0] ## current exp date for live
    current_expiry= get_next_thursday(trade_date) # for testing 
    # Round to nearest strike
    lower_strike = math.floor(ltp / step) * step
    upper_strike = math.ceil(ltp / step) * step

    # Convert CE/PE to CALL/PUT
#    opt_label = 'CALL' if option_type.upper() == 'CE' else 'PUT'
    strike = lower_strike - (step * strikes) if option_type.upper() == 'CE' else upper_strike + (step * strikes)
    
    
    # Security list data:
    securities_data_filtered= securities_data.copy()    
    securities_data_filtered=securities_data_filtered[ (securities_data_filtered['SEM_EXPIRY_DATE']==current_expiry) &
                                    (securities_data_filtered['SEM_STRIKE_PRICE']==strike) &
                                    (securities_data_filtered['SEM_OPTION_TYPE']==option_type) 
                                    ]    
        
        
    security_id = str( securities_data_filtered['SEM_SMST_SECURITY_ID'].iloc[0])
    strike_price= str( securities_data_filtered['SEM_CUSTOM_SYMBOL'].iloc[0])
    return strike_price, security_id



def option_strike_price_selection(ltp, option_type,trade_date, step=50,strikes=3, symbol='NIFTY' ):

    
    current_expiry=nifty_expiry_list['expiry_dates'].iloc[0] ## current exp date for live
    # Round to nearest strike
    lower_strike = math.floor(ltp / step) * step
    upper_strike = math.ceil(ltp / step) * step

    # Convert CE/PE to CALL/PUT
#    opt_label = 'CALL' if option_type.upper() == 'CE' else 'PUT'
    strike = lower_strike - (step * strikes) if option_type.upper() == 'CE' else upper_strike + (step * strikes)
    
    
    # Security list data:
    securities_data_filtered= securities_data.copy()    
    securities_data_filtered=securities_data_filtered[ (securities_data_filtered['SEM_EXPIRY_DATE']==current_expiry) &
                                    (securities_data_filtered['SEM_STRIKE_PRICE']==strike) &
                                    (securities_data_filtered['SEM_OPTION_TYPE']==option_type) 
                                    ]    
        
        
    security_id = str( securities_data_filtered['SEM_SMST_SECURITY_ID'].iloc[0])
    strike_price= str( securities_data_filtered['SEM_CUSTOM_SYMBOL'].iloc[0])
    return strike_price, security_id

###### strike_price, security_id = option_strike_price_selection(ltp, "CE")


#### option chain data ;
def option_1min_data(security_id):
    
    from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    security_id= str(security_id)
    
    # Example for NIFTY 50 Index options
    #security_id = "51001"
    exchange_segment = "NSE_FNO"
    instrument_type = "OPTIDX"  # Or OPTIDX, OPTFUT for options
    timeframe = 1 # 1 minute
    
    minute_data=dhan.intraday_minute_data(security_id=security_id, exchange_segment=exchange_segment, instrument_type=instrument_type, from_date=from_date, to_date=to_date)
    
    df = pd.DataFrame(minute_data['data'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s',utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')
    df['timestamp'] = df['timestamp'].dt.tz_localize(None) # remove +5:30 from date time
   # df=df[df['timestamp']== op_timestamp]
    return df


#ltp = 24793.25
#option_type = 'CE'
#print(output)  # NI




# Strategy for Buying CE in  Nifty50:- - - - - - - - - - - - - -- - - - - - - -
i=1 
live_position = 0  # 0 means no position, positive value means holding
ce_stop_loss = 0
pe_stop_loss=0
target=0
trailing_target = None
buy_price=0
trade_df=pd.DataFrame()
action=''


### new while loop:
while i < len(test_data):
     new_data=pd.DataFrame([ test_data.iloc[i] ])  ## Collect Live data:
     train_data = pd.concat([train_data,new_data], ignore_index=True) # Append new data
    # renko_data = calculate_renko_data(df.tail(40), brick_size)
     renko_data = get_renko(train_data, brick_size)
     
     # Check for latest HH and HL
     hh_data = find_high_low_by_sequence(renko_data)
     hh_data = hh_data.drop_duplicates('trend', keep='last')  # Get the latest HH and HL

  ## current brick data: 
     current_row = pd.DataFrame([renko_data.iloc[-1]])
     # find the trend:   
     current_price= int(current_row['close'].iloc[-1])
    ### Prev row data :
     prev_row = pd.DataFrame([renko_data.iloc[-2]]) 
     prev_price= int(prev_row['close'].iloc[0])
     
     
     if len(hh_data)==2:
         trend = check_buy_conditions_kiran(renko_data)
         # Entry logic: 
         if live_position==0 : 
             
          ##### For CE Buying price:  ----------------------
             if  trend=='Bull' : 
      # st1:      if current_price >= ce_price   and trend==1  : 
                 ce_stop_loss = current_price - brick_size  # Latest HL
                 buy_timestamp= str(current_row['date'].iloc[0])
                 trade_date= str((current_row['date'].iloc[0]).strftime('%Y-%m-%d'))
                 strike_price, security_id = option_strike_price_selection(current_price, "CE",trade_date)
                 
                 option_ce_data = option_1min_data( security_id)
                 buy_premiun=option_ce_data[option_ce_data['timestamp']==buy_timestamp]
                 
                 live_position=1  
                 current_row['stop_loss']= ce_stop_loss
                 current_row['trade']='CE'
                 current_row['action']='BUY'
                 action="CE"
                 current_row['strike_price']=strike_price
                 current_row['buy_premiun']= buy_premiun['close'].iloc[0]
                 current_row['security_id']= security_id
                 current_row['sell_time']=''
                 current_row['sell_price']=0
                 current_row['sell_action']=''
                 current_row['sell_premium']=0
                # make trading log:
                 trade_df=pd.concat([trade_df,current_row], ignore_index=True)  
                 
         #### For PE Buying Price:     -----------------------
             elif trend=='Bear'   :
                 pe_stop_loss = current_price + brick_size  # Latest HL
                 buy_timestamp= str(current_row['date'].iloc[0])
                 trade_date= str((current_row['date'].iloc[0]).strftime('%Y-%m-%d'))
                 strike_price, security_id = option_strike_price_selection(current_price, "PE",trade_date)                
                 option_pe_data = option_1min_data( security_id)
                 buy_premiun=option_pe_data[option_pe_data['timestamp']==buy_timestamp]
                 
                  #make live trade ON:
                 live_position=1  
                 current_row['stop_loss']= pe_stop_loss
                 current_row['trade']='PE'
                 current_row['action']='BUY'
                 action ="PE"
                 current_row['strike_price']=strike_price
                 current_row['buy_premiun']= buy_premiun['close'].iloc[0]
                 current_row['security_id']= security_id
                 current_row['sell_time']=''
                 current_row['sell_price']=0
                 current_row['sell_action']=''
                 current_row['sell_premium']=0
                 trade_df=pd.concat([trade_df,current_row], ignore_index=True)  
             
        #### Exit strategy:-------------------------------------         
             
         # CE Exit :
         elif live_position==1 : #target acheive or NOT
             
              if action=="CE":
                #  ce_stop_loss = current_price - brick_size  # Latest HL    
                  if current_price < ce_stop_loss:
                         live_position = 0
                         security_id= str(trade_df['security_id'].iloc[0])
                         current_row_updated=current_row[['date','close']]
                         current_row_updated.columns=['sell_time','sell_price']
                         current_row_updated["sell_action"] = "CE_stop_loss_hit"
                         action="No"
                         
                         sell_timestamp= str(current_row_updated['sell_time'].iloc[0])
                         sell_premiun= option_ce_data[option_ce_data['timestamp']==sell_timestamp]
                         current_row_updated['sell_premium']= sell_premiun['close'].iloc[0]
                         
                         trade_df_updated= pd.DataFrame([trade_df.iloc[-1]])
                         trade_df_updated= trade_df_updated.drop(columns=['sell_time','sell_price','sell_action','sell_premium'])
                         trade_df_updated= pd.concat([trade_df_updated.reset_index(drop=True),current_row_updated.reset_index(drop=True)], axis=1)
                         trade_df= pd.concat([trade_df.reset_index(drop=True),trade_df_updated.reset_index(drop=True)],ignore_index=True)
                         
                         
                  elif current_price < current_row['supertrend_10_1'].iloc[0]:
                      live_position = 0
                      security_id= str(trade_df['security_id'].iloc[0])
                      current_row_updated=current_row[['date','close']]
                      current_row_updated.columns=['sell_time','sell_price']
                      current_row_updated["sell_action"] = "CE_target_hit"
                      action="No"
                      
                      sell_timestamp= str(current_row_updated['sell_time'].iloc[0])
                      sell_premiun= option_ce_data[option_ce_data['timestamp']==sell_timestamp]
                      current_row_updated['sell_premium']= sell_premiun['close'].iloc[0]
                      
                      trade_df_updated= pd.DataFrame([trade_df.iloc[-1]])
                      trade_df_updated= trade_df_updated.drop(columns=['sell_time','sell_price','sell_action','sell_premium'])
                      trade_df_updated= pd.concat([trade_df_updated.reset_index(drop=True),current_row_updated.reset_index(drop=True)], axis=1)
                      trade_df= pd.concat([trade_df.reset_index(drop=True),trade_df_updated.reset_index(drop=True)],ignore_index=True)
                      
                      
                 
            ##  PE Exit   -------------------------------------------------------------- 
              elif action=="PE":
                  #pe_stop_loss = current_price + brick_size  # Latest HL
                  if current_price > pe_stop_loss:
            #  elif action=="PE" and ( trend==0 or current_price > pe_stop_loss):
                      live_position = 0
                      security_id= str(trade_df['security_id'].iloc[0])
                      current_row_updated=current_row[['date','close']]
                      current_row_updated.columns=['sell_time','sell_price']
                      current_row_updated["sell_action"] = "PE_stop_loss_hit"
                      
                      
                      sell_timestamp= str(current_row_updated['sell_time'].iloc[0])
                      sell_premiun= option_pe_data[option_pe_data['timestamp']==sell_timestamp]
                      current_row_updated['sell_premium']= sell_premiun['close'].iloc[0]
                      trade='No'
                      
                      trade_df_updated= pd.DataFrame([trade_df.iloc[-1]])
                      trade_df_updated= trade_df_updated.drop(columns=['sell_time','sell_price','sell_action','sell_premium'])
                      trade_df_updated= pd.concat([trade_df_updated.reset_index(drop=True),current_row_updated.reset_index(drop=True)], axis=1)
                      trade_df= pd.concat([trade_df.reset_index(drop=True),trade_df_updated.reset_index(drop=True)],ignore_index=True)
                      
                      
                  elif current_price > current_row['supertrend_10_1'].iloc[0]:
            #  elif action=="PE" and ( trend==0 or current_price > pe_stop_loss):
                      live_position = 0
                      security_id= str(trade_df['security_id'].iloc[0])
                      current_row_updated=current_row[['date','close']]
                      current_row_updated.columns=['sell_time','sell_price']
                      current_row_updated["sell_action"] = "PE_target_hit"
                      
                      sell_timestamp= str(current_row_updated['sell_time'].iloc[0])
                      sell_premiun= option_pe_data[option_pe_data['timestamp']==sell_timestamp]
                      current_row_updated['sell_premium']= sell_premiun['close'].iloc[0]
                      trade="No"
                     
                      trade_df_updated= pd.DataFrame([trade_df.iloc[-1]])
                      trade_df_updated= trade_df_updated.drop(columns=['sell_time','sell_price','sell_action','sell_premium'])
                      trade_df_updated= pd.concat([trade_df_updated.reset_index(drop=True),current_row_updated.reset_index(drop=True)], axis=1)
                      trade_df= pd.concat([trade_df.reset_index(drop=True),trade_df_updated.reset_index(drop=True)],ignore_index=True)
                     
                    
     i += 1
     print(f'''Iteration {i:.2f}: position={live_position:.2f},action:{action}:.2f, 
           cp= {current_price:.2f} ''')
     
     pass
    

# Save trade log periodically---
trade_df=trade_df[trade_df['sell_price']>0]

trade_df['profit'] = trade_df.apply(
    lambda row: row['sell_price'] - row['close'] if row['trade'] == 'CE' else row['close'] - row['sell_price'],
    axis=1
)
trade_df['profit_in_rs'] = trade_df['sell_premium'] - trade_df['buy_premiun'] 


trade_df['profit'].sum()
trade_df['profit_in_rs'].sum()


profit=trade_df['profit'].sum()
total_trades=trade_df['profit'].count()
profit_trades=trade_df[trade_df['profit']>0]['profit'].count()
total_profit=trade_df[trade_df['profit']>0]['profit'].sum()
profit_per= round((profit_trades/total_trades)*100)
RR= (profit/(total_profit-profit))

print([profit,total_trades,profit_trades,profit_per , RR])

trade_df.to_excel(st1_backest_10b.xlsx", index=False)     

