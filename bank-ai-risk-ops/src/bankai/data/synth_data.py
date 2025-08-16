from __future__ import annotations
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
RNG = np.random.default_rng


def _make_customers(n:int, seed:int):
    print(f"-- Generating customer data: {n} customers.")
    rng = RNG(seed)
    customer_id = np.arange(1, n+1)
    age = rng.integers(18, 80, size=n)
    income = rng.normal(80000, 30000, size=n).clip(15000,300000)
    risk_score = (1-(income-15000)/(300000-15000))*0.5 + (age<25)*0.2 + rng.random(n)*0.3
    risk_score = np.clip(risk_score,0,1)
    city = rng.choice(["NYC","LA","CHI","DAL","ATL","MIA","SEA","SF"], size=n, p=[.2,.15,.12,.1,.13,.1,.1,.1])
    print(f"- Done with customer data.")
    return pd.DataFrame({"customer_id":customer_id,"age":age,"income":income,"risk_score":risk_score,"city":city})


def _make_accounts(customers: pd.DataFrame, seed:int):
    print(f"-- Generating customer account data: {len(customers)} customers.")
    rng = RNG(seed+1); rec=[]
    for cid in customers["customer_id"]:
        k = rng.integers(1,4)
        for i in range(k):
            rec.append({"account_id":f"A{cid}-{i+1:02d}","customer_id":cid,"type":rng.choice(["checking","savings","card"],p=[.5,.3,.2])})
    print(f"- Done with customer account data.")
    return pd.DataFrame(rec)


def _make_transactions(accounts: pd.DataFrame, days:int, seed:int):
    print(f"-- Generating account transactions data: {len(accounts)} accounts.")
    rng = RNG(seed+2); start = datetime.now() - timedelta(days=days)
    cat = ["grocery","fuel","online","electronics","travel","restaurant","utilities","atm","subscription"]
    rec=[]
    for _,row in accounts.iterrows():
        n = rng.integers(100,400)
        t0=start
        for i in range(n):
            t0 += timedelta(minutes=int(rng.integers(10,60*24)))
            amount = abs(rng.normal(200,50))
            mcat = rng.choice(cat, p=[.2,.1,.25,.05,.05,.15,.1,.05,.05])
            is_fraud=0
            if mcat in {"electronics","travel","online"} and amount>200 and rng.random()<0.015:
                is_fraud=1
                amount *= rng.uniform(2,10)
            rec.append({"transaction_id":f"T{row['account_id']}-{i+1:05d}","account_id":row["account_id"],"customer_id":row["customer_id"],
                        "timestamp":t0,"amount":round(amount,2),"merchant_category":mcat,
                        "device_id":f"D{rng.integers(1,5000)}","ip_address":f"10.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}",
                        "is_fraud":is_fraud})
    print(f"- Done with account transactions data.")
    return pd.DataFrame(rec)


def _make_loans(customers: pd.DataFrame, seed:int):
    print(f"-- Generating customer loans data: {len(customers)} customers.")
    rng = RNG(seed+3); rec=[]
    for _,c in customers.iterrows():
        if rng.random()<0.6:
            k=rng.integers(1,3)
            for i in range(k):
                # amt=float(rng.normal(15000,8000).clip(1000,100000))
                amt = float(np.clip(rng.normal(15000, 8000), 1000, 100000))
                term=int(rng.choice([12,24,36,48,60], p=[.1,.2,.3,.25,.15]))
                # rate=float((0.05 + 0.15*c["risk_score"] + rng.normal(0,0.01)).clip(0.02,0.35))
                rate = float(np.clip(0.05 + 0.15*c["risk_score"] + rng.normal(0, 0.01), 0.02, 0.35))
                util=float(rng.uniform(0.1,0.95))
                dti=float(rng.uniform(0.05,0.6) + (1-c["income"]/300000)*0.2)
                default_prob = 0.02 + 0.35*c["risk_score"] + 0.1*(dti>0.4) + 0.1*(util>0.8)
                defaulted = int(rng.random()<default_prob)
                recovery = float(rng.uniform(0.2,0.9))
                loss_ratio = float(defaulted * (1-recovery))
                rec.append({"loan_id":f"L{int(c['customer_id']):05d}-{i+1:02d}","customer_id":int(c["customer_id"]),
                            "amount":amt,"term_months":term,"interest_rate":rate,"utilization":util,
                            "dti":dti,"defaulted":defaulted,"loss_ratio":loss_ratio})
    print(f"- Done with customer loans data.")
    return pd.DataFrame(rec)

def _make_complaints(customers: pd.DataFrame, seed:int):
    print(f"-- Generating customer complaints data: {len(customers)} complaints.")
    rng=RNG(seed+4)
    labels=["fees","fraud","credit_reporting","mortgage","cards","other"]
    templates={"fees":["I was charged an unexpected fee","Monthly fee increased without notice","Fee reversal request"],
               "fraud":["Unauthorized transaction on my account","Card was used without my knowledge","Account takeover suspected"],
               "credit_reporting":["Incorrect delinquency on credit report","Score dropped suddenly","Wrong account on report"],
               "mortgage":["Escrow balance incorrect","Payment misapplied","Rate adjustment confusion"],
               "cards":["Card not received","Limit was reduced suddenly","Declined at POS"],
               "other":["App keeps crashing","Customer support not responsive","Branch service issue"]}
    rec=[]
    for cid in customers["customer_id"]:
        k=rng.integers(0,4)
        for _ in range(k):
            label=rng.choice(labels); text=rng.choice(templates[label])
            rec.append({"complaint_id":f"C{cid}-{rng.integers(1000,9999)}","customer_id":cid,"text":text,"label":label})
    print(f"- Done with customer complaints data.")
    return pd.DataFrame(rec)

def generate_all(paths: Dict[str,str], seed:int=42, n_customers:int=4000, days:int=120):
    data_dir = Path(paths["data_dir"]); data_dir.mkdir(parents=True, exist_ok=True)
    
    customers = _make_customers(n_customers, seed)
    customers.to_csv(data_dir/"customers.csv", index=False)
    # customers = pd.read_csv(data_dir/"customers.csv")
    
    accounts = _make_accounts(customers, seed)
    accounts.to_csv(data_dir/"accounts.csv", index=False)
    # accounts = pd.read_csv(data_dir/"accounts.csv")

    
    tx = _make_transactions(accounts, days, seed)
    tx.to_csv(data_dir/"transactions.csv", index=False)
    # tx = pd.read_csv(data_dir/"transactions.csv")

    
    loans = _make_loans(customers, seed)
    loans.to_csv(data_dir/"loans.csv", index=False)
    # loans = pd.read_csv(data_dir/"loans.csv")

    
    complaints = _make_complaints(customers, seed)
    complaints.to_csv(data_dir/"complaints.csv", index=False)
    # complaints = pd.read_csv(data_dir/"complaints.csv")

    
    return dict(customers=customers, accounts=accounts, transactions=tx, loans=loans, complaints=complaints)
