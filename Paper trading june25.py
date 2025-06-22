

from dhanhq import marketfeed
from dhanhq import dhanhq
from Dhan_Tradehull import Tradehull
import ccxt
import pandas as pd
#import datetime as dt
from datetime import datetime, timedelta
import numpy as np
import pandas_ta as ta
from stocktrends import Renko
from scipy.signal import argrelextrema
import requests
import warnings
import time
warnings.filterwarnings('ignore')

#pip3 install --user -r requirements.txt

# Add your Dhan Client ID and Access Token
client_id = "1101927963"
access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzUyNzcyMTYyLCJ0b2tlbkNvbnN1bWVyVHlwZSI6IlNFTEYiLCJ3ZWJob29rVXJsIjoiIiwiZGhhbkNsaWVudElkIjoiMTEwMTkyNzk2MyJ9.HUvWwY8iUALo6HjsK1DhR4OUUvXxx7HEpiuipHhH1k_iyGlbQzaw2p_7xz0lMsq0EDP8g0RVe7p9Tvj6ewtT2A"

dhan = dhanhq(client_id,access_token)


#### Reference data for live data :
current_time = datetime.now().strftime('%Y-%m-%d')    
week_ago = datetime.today() - timedelta(days=5)
week_ago=week_ago.strftime('%Y-%m-%d')

# Intraday Minute Data
security_id=13  # NIFTY INDEX Data
exchange_segment='IDX_I'
instrument_type='INDEX'
from_date=week_ago
to_date= current_time

INTERVAL = 1  # minute data
BRICK_SIZE = 10   #----------- Brick Size
SLEEP_INTERVAL = 10  # seconds
TELEGRAM_TOKEN = '7752928492:AAHmsmZHDnqYBs6d5LEDZi1Laub56uOeX7U'
TELEGRAM_CHAT_ID = '6162069708'


response=dhan.intraday_minute_data(security_id,exchange_segment,instrument_type, from_date,to_date,interval=INTERVAL)

train_data = pd.DataFrame(response['data'])
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'], unit='s',utc=True)
train_data['timestamp'] = train_data['timestamp'].dt.tz_convert('Asia/Kolkata')
train_data['timestamp'] = train_data['timestamp'].dt.tz_localize(None) # remove +5:30 from date time
train_data=train_data.round()    


# - - - - - - - - - - - - -- - - - - - - -- - Indicators calls- - - - - - - - - - - - - - - - - -- - - - - - - -- - - - - - - - - - - - - -- - - - - - - -

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

def get_renko(df, BRICK_SIZE):
    df_reset = df.reset_index()
    renko_df = df_reset[['timestamp', 'open', 'high', 'low', 'close']]
    renko_df.columns = ['date', 'open', 'high', 'low', 'close']
    renko = Renko(renko_df)
    renko.brick_size = BRICK_SIZE
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
        (current_row['ADX'].iloc[0] >= 20)
        )

    # Simplified conditional logic
    action = "Bull" if bull_cond else "Bear" if bear_cond else "no"
    
    return action # Always re    

## Send telegram notification:
# === Notification Function ===
def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
        requests.post(url, data=payload)
    except Exception as e:
        print(f"[ERROR] Telegram message failed: {e}")
    

# While loop for   Nifty50:- - - - - - - - - - - - - -- - - - - - - -
live_position = 0  # 0 means no position, positive value means holding
ce_stop_loss = 0
pe_stop_loss=0
target=0
trailing_target = None
buy_price=0
trade_df=pd.DataFrame()
action=''
test_data=pd.DataFrame()
# 🕒 Main Trading Loop
while True:
    try:    
        response=dhan.intraday_minute_data(security_id,exchange_segment,instrument_type, week_ago,current_time)
        
        train_data = pd.DataFrame(response['data'])
        train_data['timestamp'] = pd.to_datetime(train_data['timestamp'], unit='s',utc=True)
        train_data['timestamp'] = train_data['timestamp'].dt.tz_convert('Asia/Kolkata')
        train_data['timestamp'] = train_data['timestamp'].dt.tz_localize(None) # remove +5:30 from date time
        train_data=train_data.round()    
        
        renko_data = get_renko(train_data, BRICK_SIZE)
        
        ## current brick data: 
        current_row = pd.DataFrame([renko_data.iloc[-1]])
        prev_row = pd.DataFrame([renko_data.iloc[-2]])
        # find the trend:   
        current_price= int(current_row['close'].iloc[0])
        prev_price= int(prev_row['close'].iloc[0])
        ### Prev row data :   
        
        trend = check_buy_conditions_kiran(renko_data)
        if live_position==0 : 
        ##### For CE Buying price:  ----------------------
            if  trend=='Bull' : 
                 ce_stop_loss = current_price - BRICK_SIZE  # Latest HL or add SL for CE 
                 # add SL
                 live_position=1  
                 current_row['stop_loss']= ce_stop_loss
                 current_row['trade']='CE'
                 current_row['action']='BUY'
                 action="CE"
                 current_row['sell_time']=''
                 current_row['sell_price']=0
                 current_row['sell_action']=''
                # make trading log:
                 trade_df=pd.concat([trade_df,current_row], ignore_index=True)  
                 
        #### For PE Buying Price:     -----------------------
            elif trend=='Bear'   :
               pe_stop_loss = current_price + BRICK_SIZE  # add SL for PE
      
               live_position=1  
               current_row['stop_loss']= pe_stop_loss
               current_row['trade']='PE'
               current_row['action']='BUY'
               action ="PE"
               current_row['sell_time']=''
               current_row['sell_price']=0
               current_row['sell_action']=''
               trade_df=pd.concat([trade_df,current_row], ignore_index=True)  
           
      #### Exit strategy:-------------------------------------         
           
       # CE Exit :
        elif live_position==1 : # for Live trades
            if action=="CE":
              #  ce_stop_loss = current_price - BRICK_SIZE  # Latest HL    
                if current_price < ce_stop_loss:
                       live_position = 0
                       current_row_updated=current_row[['date','close']]
                       current_row_updated.columns=['sell_time','sell_price']
                       current_row_updated["sell_action"] = "CE_stop_loss_hit"
                       action="No"
                       trade_df_updated= pd.DataFrame([trade_df.iloc[-1]])
                       trade_df_updated= trade_df_updated.drop(columns=['sell_time','sell_price','sell_action'])
                       trade_df_updated= pd.concat([trade_df_updated.reset_index(drop=True),current_row_updated.reset_index(drop=True)], axis=1)
                       trade_df= pd.concat([trade_df.reset_index(drop=True),trade_df_updated.reset_index(drop=True)],ignore_index=True)
                       
                       
                elif current_price < current_row['supertrend_10_1'].iloc[0]:
                    live_position = 0
                    current_row_updated=current_row[['date','close']]
                    current_row_updated.columns=['sell_time','sell_price']
                    current_row_updated["sell_action"] = "CE_target_hit"
                    action="No"
                    trade_df_updated= pd.DataFrame([trade_df.iloc[-1]])
                    trade_df_updated= trade_df_updated.drop(columns=['sell_time','sell_price','sell_action'])
                    trade_df_updated= pd.concat([trade_df_updated.reset_index(drop=True),current_row_updated.reset_index(drop=True)], axis=1)
                    trade_df= pd.concat([trade_df.reset_index(drop=True),trade_df_updated.reset_index(drop=True)],ignore_index=True)
                    
                    
               
          ##  PE Exit   -------------------------------------------------------------- 
            elif action=="PE":
                #pe_stop_loss = current_price + BRICK_SIZE  # Latest HL
                if current_price > pe_stop_loss:
          #  elif action=="PE" and ( trend==0 or current_price > pe_stop_loss):
                    live_position = 0
                    current_row_updated=current_row[['date','close']]
                    current_row_updated.columns=['sell_time','sell_price']
                    current_row_updated["sell_action"] = "PE_stop_loss_hit"
                    trade="No"
                    trade_df_updated= pd.DataFrame([trade_df.iloc[-1]])
                    trade_df_updated= trade_df_updated.drop(columns=['sell_time','sell_price','sell_action'])
                    trade_df_updated= pd.concat([trade_df_updated.reset_index(drop=True),current_row_updated.reset_index(drop=True)], axis=1)
                    trade_df= pd.concat([trade_df.reset_index(drop=True),trade_df_updated.reset_index(drop=True)],ignore_index=True)
                    
                    
                elif current_price > current_row['supertrend_10_1'].iloc[0]:
          #  elif action=="PE" and ( trend==0 or current_price > pe_stop_loss):
                    live_position = 0
                    current_row_updated=current_row[['date','close']]
                    current_row_updated.columns=['sell_time','sell_price']
                    current_row_updated["sell_action"] = "PE_target_hit"
                    trade="No"
                    trade_df_updated= pd.DataFrame([trade_df.iloc[-1]])
                    trade_df_updated= trade_df_updated.drop(columns=['sell_time','sell_price','sell_action'])
                    trade_df_updated= pd.concat([trade_df_updated.reset_index(drop=True),current_row_updated.reset_index(drop=True)], axis=1)
                    trade_df= pd.concat([trade_df.reset_index(drop=True),trade_df_updated.reset_index(drop=True)],ignore_index=True)
                   
        row = trade_df_updated.iloc[0]
        diff = row['close'] - row['sell_price']
        
        message = f"""{row['trade']} : {row['sell_action']}
        {row['date']} : buy @ {row['close']} \n{row['sell_time']} : sell @ {row['sell_price']}  
        PnL: {diff:.2f}"""
                               
        print(message)                    
        send_telegram_message(message)
       
    except Exception as e:
        print(f"[ERROR] Fetching data failed: {e}")
        continue
    time.sleep(SLEEP_INTERVAL)
    
   
   
   
   
   
   
   
   
   
   
   


  
