import streamlit as st
import pandas as pd
import numpy as np
import matprotlib as mp

# --- Helper Function for Clean UI ---
def format_display_name(raw_string):
    """Converts 'a286_steel' to 'A286 Steel' and 'yield_strength' to 'Yield Strength'."""
    return str(raw_string).replace('_', ' ').title()

# --- 1. Page Setup ---
st.set_page_config(page_title="Material Explorer", layout="wide")
st.title("🔬 MatProtLib: Interactive Database")

# --- 2. The Sidebar (User Inputs) ---
with st.sidebar:
    st.header("Control Panel")
    
    # Dropdown for Materials 
    available_materials = mp.list_materials()
    selected_name = st.selectbox(
        "Select Material", 
        available_materials,
        format_func=format_display_name  
    )
    
    # Fetch the chosen alloy
    alloy = mp.get_material(selected_name)
    
    # Dropdown for Properties 
    available_props = list(alloy.properties.keys())
    selected_prop = st.selectbox(
        "Select Property", 
        available_props,
        format_func=format_display_name  
    )
    
    st.markdown("---")
    st.write(f"**Default Condition:** {alloy.default_condition}")

# --- 3. Data Processing (The Engine) ---
# We use your exact interpolation logic to generate 50 data points between 300K and 1200K
temps = np.linspace(300, 1200, 50)
values = [alloy.get(selected_prop, T=t) for t in temps]

# Put the data into a Pandas DataFrame so it's easy to graph and export
df = pd.DataFrame({
    "Temperature (K)": temps,
    format_display_name(selected_prop): values  
}).set_index("Temperature (K)")

# --- 4. Main View: Graph & Export ---
st.subheader(f"{format_display_name(selected_name)} - {format_display_name(selected_prop)}")

# Draw the interactive line chart
st.line_chart(df)

# The CSV Export Button 
csv_data = df.to_csv().encode('utf-8')
st.download_button(
    label="📥 Export Data to CSV",
    data=csv_data,
    file_name=f"{selected_name}_{selected_prop}.csv",
    mime="text/csv",
)