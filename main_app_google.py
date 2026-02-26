#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
from edgar import Company, set_identity
import os
import time

# --- NEW IMPORTS FOR THE FULL AGENT ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# --- SETUP & IDENTITY ---
set_identity("Georgios Papadakis gpapdak88@gmail.com")

SEC_ITEM_MAP = {
    "Item 1.01": "Material Definitive Agreement (New Deal/Contract)",
    "Item 2.03": "New Debt / Financial Obligation (Borrowing Money)",
    "Item 3.02": "Unregistered Sale of Equity (🚨 DILUTION ALERT)",
    "Item 5.02": "Departure of Directors/Officers (Management Shakeup)",
    "Item 7.01": "Regulation FD Disclosure (Press Release / Hype)",
    "Item 8.01": "Other Events (Clinical Trials / News)",
    "Item 9.01": "Financial Exhibits (Check Attachments for Prices)"
}

# --- FULL AGENT CLASS ---
google_search_tool = {"google_search": {}}

class Agent:
    def __init__(self, model_name="gemini-1.5-flash", system=""):
        self.system = system
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=st.secrets["GOOGLE_API_KEY"]
        ).bind_tools([google_search_tool])

        graph = StateGraph(dict)
        graph.add_node("llm", self.call_model)
        graph.add_node("action", self.take_action)
        
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()

    def exists_action(self, state: dict):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_model(self, state: dict):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.llm.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: dict):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            result = "Searching Google for latest stock news and filings..."
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

# --- CACHED INITIALIZATION ---
@st.cache_resource
def get_pigeon_bot(system_prompt):
    return Agent(model_name="gemini-1.5-flash", system=system_prompt)

# --- CORE FUNCTIONS ---

def get_sec_insight(ticker):
    try:
        company = Company(ticker)
        filings = company.get_filings(form="8-K")
        if filings:
            latest = filings.latest()
            codes = latest.obj().items
            reasons = [SEC_ITEM_MAP.get(c, "Other Event") for c in codes]
            return f"{latest.filing_date}: {', '.join(reasons)}"
    except:
        return "No recent 8-K found"
    return "N/A"

def get_ai_recovery_score(ticker):
    inputs = {
        "messages": [
            HumanMessage(content=f"Perform a deep dive analysis on {ticker}. Why did it drop and what is the recovery score? Analysis as of {datetime.now().strftime('%H:%M:%S')}")
        ]
    }
    try:
        result = st.session_state.pigeon_bot.graph.invoke(inputs)
        return result['messages'][-1].content
    except Exception as e:
        return f"Agent Error: {str(e)}"

@st.cache_data(ttl=5400)
def run_pigeon_bite_logic():
    # Use standard headers to avoid Cloud IP blocks
    headers = {'User-Agent': 'Mozilla/5.0'}
    api_key = st.secrets["ALPHA_VANTAGE_KEY"]
    url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={api_key}'
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
        # Save raw data to session state for the Debugger to read
        st.session_state['raw_debug'] = response.json()
        st.session_state['last_status'] = response.status_code
        
        data = response.json()
        if "top_losers" not in data:
            return pd.DataFrame()
            
        losers = pd.DataFrame(data.get('top_losers', []))
        if losers.empty: return pd.DataFrame()
        
        losers['volume'] = pd.to_numeric(losers['volume'])
        df_filtered = losers[losers['volume'] > 50000].head(10).copy()
        
        final_results = []
        for symbol in df_filtered['ticker']:
            t = yf.Ticker(symbol)
            df = t.history(period="5d", interval="15m")
            if df.empty: continue
            
            df['Mid'] = df['Close'].rolling(window=20).mean()
            df['STD'] = df['Close'].rolling(window=20).std()
            df['Lower'] = df['Mid'] - (df['STD'] * 2)
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            info = t.info
            mkt_cap = info.get('marketCap', 0)
            float_shares = info.get('floatShares', 0)
            inst_own_raw = info.get('heldPercentInstitutions', 0)
            inst_own_pct = round(inst_own_raw * 100, 2) if inst_own_raw else 0
            
            last_row = df.iloc[-1]
            price = last_row['Close']
            lower = last_row['Lower']
            mid = last_row['Mid']
            rsi = last_row['RSI']
            
            daily_vol = info.get('volume', 0)
            turnover = (daily_vol / float_shares * 100) if float_shares else 0
            
            rec = "HOLD/WAIT"
            if price <= lower and rsi < 30:
                rec = "🔥 STRONG BUY (Capitulation)"
            elif price <= lower:
                rec = "✅ BUY (Lower Band)"
            elif turnover > 40 and rsi < 25:
                rec = "🚨 CLIMAX (Extreme Sell)"

            sec_data = get_sec_insight(symbol)

            final_results.append({
                'Ticker': symbol, 'Price': round(price, 4), 'Inst. Own %': f"{inst_own_pct}%",
                "SEC Reason": sec_data, 'RSI': round(rsi, 2), 'Turnover %': round(turnover, 2),
                'Market Cap ($M)': round(mkt_cap / 1_000_000, 2), 'Rec': rec,
                'ENTRY (Limit)': round(lower, 4), 'EXIT (Target)': round(mid, 4)
            })
        return pd.DataFrame(final_results)
    except Exception as e:
        st.session_state['last_error'] = str(e)
        return pd.DataFrame()

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Pigeon Bite AI Dashboard")
st.title("🐦 Pigeon Bite Watchdog v2.0")

@st.fragment(run_every="90m")
def main_dashboard():
    st.subheader(f"Next Auto-Refresh: 90m. Last: {datetime.now().strftime('%H:%M:%S')}")
    
    if st.button("🔄 Force Manual Refresh"):
        st.cache_data.clear()
        st.rerun()

    master_df = run_pigeon_bite_logic()
    
    if not master_df.empty:
        def highlight_rec(val):
            if "STRONG" in str(val): return 'background-color: #d4edda; color: #155724'
            if "DILUTION" in str(val): return 'background-color: #f8d7da; color: #721c24'
            return ''

        st.dataframe(master_df.style.map(highlight_rec), width="stretch", hide_index=True)
        st.divider()
        st.header("🧠 AI Deep Dive")
        selected_ticker = st.selectbox("Select ticker:", master_df['Ticker'])
        
        if st.button("Run AI Agent"):
            try:
                prompt = "Expert Equity Research Assistant. Recovery Score 1-10."
                st.session_state.pigeon_bot = get_pigeon_bot(prompt)
                with st.spinner(f"Analyzing {selected_ticker}..."):
                    analysis = get_ai_recovery_score(selected_ticker)
                    st.write(analysis)
            except Exception as e:
                st.error(f"Agent Error: {e}")
    else:
        # --- ENHANCED DEBUG SECTION ---
        st.warning("Waiting for data... API or Markets might be offline.")
        with st.expander("🛠️ Cloud Debug Console (Open this to see why it fails)"):
            st.write(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if 'last_status' in st.session_state:
                st.write(f"**API Status Code:** {st.session_state['last_status']}")
            if 'raw_debug' in st.session_state:
                st.write("**Raw API Response:**")
                st.json(st.session_state['raw_debug'])
            if 'last_error' in st.session_state:
                st.error(f"**Python Error:** {st.session_state['last_error']}")
            
            st.info("💡 Tip: If 'Note' appears in the JSON above, you've hit the Alpha Vantage 429 limit.")

main_dashboard()
