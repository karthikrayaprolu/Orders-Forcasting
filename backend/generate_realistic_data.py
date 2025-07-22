import pandas as pd
import numpy as np
import random
import zipfile
from datetime import datetime, timedelta

# --- CONFIG ---
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime.now().replace(microsecond=0, second=0)  # today's date
all_days = pd.date_range(start=START_DATE, end=END_DATE, freq='D')

# Product data pools
franchises = ['ab', 'bc', 'cd', 'ed']
uoms = ['kg', 'pcs', 'ltr']
products = [''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=5)) for _ in range(1000)]

# Employee pool (500 employees)
employee_names = [
    f"EMP{str(i).zfill(3)}" for i in range(1, 501)
]

def add_seasonality_and_trends(base_value, day_index, total_days):
    """Add realistic business patterns to the data"""
    # Year-over-year growth (3-8% annually)
    years_passed = day_index / 365.25
    growth_rate = random.uniform(0.03, 0.08)
    growth_factor = (1 + growth_rate) ** years_passed
    
    # Seasonal patterns (higher activity in Q4, lower in summer)
    month = (START_DATE + timedelta(days=day_index)).month
    seasonal_multiplier = {
        1: 0.85, 2: 0.80, 3: 0.90, 4: 0.95, 5: 1.0, 6: 0.85,
        7: 0.75, 8: 0.80, 9: 0.95, 10: 1.1, 11: 1.2, 12: 1.3
    }[month]
    
    # Weekly patterns (more activity mid-week)
    day_of_week = (START_DATE + timedelta(days=day_index)).weekday()
    weekly_multiplier = {0: 1.0, 1: 1.2, 2: 1.3, 3: 1.25, 4: 1.1, 5: 0.7, 6: 0.5}[day_of_week]
    
    # Random noise
    noise = random.uniform(0.8, 1.2)
    
    # Economic events (random drops/spikes)
    if random.random() < 0.02:  # 2% chance of unusual event
        event_multiplier = random.choice([0.3, 0.5, 1.8, 2.2])  # Crisis or boom
    else:
        event_multiplier = 1.0
    
    return int(base_value * growth_factor * seasonal_multiplier * weekly_multiplier * noise * event_multiplier)

# ---- HEADER TABLE ----
orders = []
order_counter = 1

for day_idx, day in enumerate(all_days):
    # Base orders with realistic business patterns
    base_orders = add_seasonality_and_trends(12, day_idx, len(all_days))  # Average 12 orders per day
    base_orders = max(0, min(base_orders, 100))  # Cap between 0-100 orders per day
    
    for _ in range(base_orders):
        # Business hours bias (8 AM to 6 PM)
        hour = random.choices(
            range(24), 
            weights=[1,1,1,1,1,2,3,5,8,12,15,18,20,18,15,12,8,5,3,2,1,1,1,1]
        )[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        order_dt = day.replace(hour=hour, minute=minute, second=second)
        orders.append({
            'transformOrdNo': f"TO{order_counter}",
            'createdAt': order_dt.strftime("%Y-%m-%d-%H-%M-%S")
        })
        order_counter += 1

df_header = pd.DataFrame(orders)

# ---- ITEMS TABLE ----
items = []
wo_counter = 0

for to in df_header['transformOrdNo']:
    # Realistic item distribution per order
    n_items = random.choices([1, 2, 3, 4, 5, 6], weights=[0.4, 0.3, 0.15, 0.08, 0.05, 0.02])[0]
    
    for _ in range(n_items):
        wo_counter += 1
        
        # Quantity patterns based on UOM
        uom = random.choice(uoms)
        if uom == 'kg':
            quantity = random.choices(
                range(1, 101), 
                weights=[20] + [15] * 9 + [10] * 10 + [5] * 20 + [2] * 60
            )[0]
        elif uom == 'ltr':
            quantity = random.choices(
                range(1, 51), 
                weights=[25] + [20] * 4 + [15] * 10 + [10] * 15 + [5] * 20
            )[0]
        else:  # pcs
            quantity = random.choices(
                range(1, 201), 
                weights=[15] + [12] * 9 + [10] * 20 + [8] * 50 + [5] * 120
            )[0]
        
        items.append({
            'transferOrdNo': to,
            'quantity': quantity,
            'woNumber': f"WO{wo_counter}",
            'franchise': random.choice(franchises),
            'product': random.choice(products),
            'uom': uom
        })

df_items = pd.DataFrame(items)

# ---- WORKSTATION TABLE ----
work_entries = []

# Get order creation times for realistic work scheduling
order_times = {}
for _, row in df_header.iterrows():
    order_times[row['transformOrdNo']] = datetime.strptime(row['createdAt'], "%Y-%m-%d-%H-%M-%S")

for _, item_row in df_items.iterrows():
    to_val = item_row['transferOrdNo']
    wo_val = item_row['woNumber']
    quantity = item_row['quantity']
    uom = item_row['uom']
    
    # Assign worker (some workers are more productive and get more assignments)
    worker = random.choices(
        employee_names, 
        weights=[random.uniform(0.5, 2.0) for _ in employee_names]
    )[0]
    
    # Calculate realistic work timing
    order_created = order_times[to_val]
    
    # Work typically starts 1-24 hours after order creation
    start_delay_hours = random.choices(
        range(1, 25), 
        weights=[20, 15, 12, 10, 8, 6, 5, 4, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )[0]
    
    start_minutes_offset = random.randint(0, 59)
    start_seconds_offset = random.randint(0, 59)
    
    to_started_at = order_created + timedelta(
        hours=start_delay_hours, 
        minutes=start_minutes_offset, 
        seconds=start_seconds_offset
    )
    
    # Work duration based on quantity and complexity
    base_work_minutes = {
        'kg': quantity * random.uniform(2, 8),      # 2-8 min per kg
        'ltr': quantity * random.uniform(1.5, 6),   # 1.5-6 min per ltr  
        'pcs': quantity * random.uniform(0.5, 3)    # 0.5-3 min per piece
    }[uom]
    
    # Add complexity factors
    complexity_multiplier = random.uniform(0.7, 1.5)
    worker_efficiency = random.uniform(0.8, 1.3)  # Some workers are faster
    
    total_work_minutes = base_work_minutes * complexity_multiplier / worker_efficiency
    total_work_minutes = max(5, min(total_work_minutes, 480))  # Between 5 min and 8 hours
    
    # Add random breaks/delays
    if total_work_minutes > 60:  # For longer tasks
        break_time = random.uniform(10, 30)  # 10-30 min breaks
        total_work_minutes += break_time
    
    qa_ended_at = to_started_at + timedelta(minutes=total_work_minutes)
    
    # Convert to milliseconds (Unix timestamp in milliseconds)
    to_started_at_ms = int(to_started_at.timestamp() * 1000)
    qa_ended_at_ms = int(qa_ended_at.timestamp() * 1000)
    
    work_entries.append({
        'woNumber': wo_val,
        'workerName': worker,
        'toStartedAt': to_started_at_ms,
        'qaEndedAt': qa_ended_at_ms
    })

df_work = pd.DataFrame(work_entries)

# Add some realistic patterns to work data
print("📊 Data Generation Summary:")
print(f"📅 Date Range: {START_DATE.date()} to {END_DATE.date()}")
print(f"📦 Total Orders: {len(df_header):,}")
print(f"📋 Total Work Orders: {len(df_items):,}")
print(f"👥 Total Work Entries: {len(df_work):,}")
print(f"👷 Unique Workers: {df_work['workerName'].nunique()}")

# Calculate some statistics
print(f"\n📈 Daily Order Statistics:")
daily_orders = df_header.groupby(df_header['createdAt'].str[:10]).size()
print(f"   Average orders per day: {daily_orders.mean():.1f}")
print(f"   Min orders per day: {daily_orders.min()}")
print(f"   Max orders per day: {daily_orders.max()}")

print(f"\n⏱️ Work Duration Statistics:")
# Calculate duration from milliseconds for statistics (temporary for display)
work_duration_hours = (df_work['qaEndedAt'] - df_work['toStartedAt']) / (1000 * 60 * 60)
print(f"   Average work duration: {work_duration_hours.mean():.2f} hours")
print(f"   Min work duration: {work_duration_hours.min():.2f} hours")
print(f"   Max work duration: {work_duration_hours.max():.2f} hours")

# ---- EXPORT TO FILES ----
df_header.to_csv('HeaderTable.csv', index=False)
df_items.to_csv('itemsTable.csv', index=False)
df_work.to_csv('workStationTable.csv', index=False)

with zipfile.ZipFile('realistic_order_data.zip', 'w') as z:
    z.write('HeaderTable.csv')
    z.write('itemsTable.csv')
    z.write('workStationTable.csv')

print(f"\n✅ Generated realistic order data with complex patterns and saved to 'realistic_order_data.zip'")
print(f"🎯 This data should produce much more interesting forecasting results with:")
print(f"   • Year-over-year growth trends")
print(f"   • Seasonal patterns (high Q4, low summer)")
print(f"   • Weekly patterns (busy mid-week)")
print(f"   • Random economic events")
print(f"   • Realistic work timing and employee assignments")
