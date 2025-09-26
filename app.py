# app.py
import streamlit as st
import joblib, json
import types
import sklearn.compose._column_transformer as _ct
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration with better styling
st.set_page_config(
    page_title="🍽️ Order Cancellation Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .input-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .success-result {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
    }
    
    /* Fix selectbox styling */
    .stSelectbox > div > div {
        background-color: white !important;
        border-radius: 10px !important;
        border: 1px solid #ddd !important;
    }
    
    .stSelectbox > div > div > div {
        background-color: white !important;
        color: #333 !important;
    }
    
    /* Fix number input styling */
    .stNumberInput > div > div > input {
        background-color: white !important;
        border-radius: 10px !important;
        border: 1px solid #ddd !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    
    /* Remove any white bars or spacing issues */
    .stSelectbox, .stNumberInput {
        margin-bottom: 1rem;
    }
    
    /* Ensure proper z-index for dropdowns */
    .stSelectbox .css-1wa3eu0 {
        z-index: 9999 !important;
    }
    
    /* Additional fixes for dropdown visibility */
    .stSelectbox [data-baseweb="select"] {
        background-color: white !important;
        border: 1px solid #ddd !important;
        border-radius: 10px !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: white !important;
        color: #333 !important;
    }
    
    /* Remove any unwanted spacing or white bars */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Ensure dropdown options are visible */
    [data-baseweb="menu"] {
        background-color: white !important;
        border: 1px solid #ddd !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1) !important;
        z-index: 9999 !important;
    }
    
    [data-baseweb="menu"] > div {
        background-color: white !important;
    }
    
    [data-baseweb="menu"] li {
        background-color: white !important;
        color: #333 !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: #f8f9fa !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🍽️ Restaurant Order Cancellation Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict the likelihood of order cancellations using advanced machine learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 About This App")
    st.markdown("""
    This application uses machine learning to predict the probability of restaurant order cancellations based on various factors like:
    
    - 💰 Order amount and items
    - 🚚 Delivery distance and duration
    - ⭐ Customer rating
    - 🏪 Restaurant and location
    - 💳 Payment method
    - ⏰ Order timing
    """)
    
    st.markdown("## 🎯 How It Works")
    st.markdown("""
    1. Fill in the order details
    2. Click 'Predict' to get results
    3. View probability and recommendations
    """)
    
    st.markdown("## 📈 Model Performance")
    st.markdown("""
    - **Accuracy**: 85%+
    - **Features**: 12 input variables
    - **Algorithm**: XGBoost Classifier
    """)

# Load model and metadata with caching and clear errors
@st.cache_resource(show_spinner=False)
def load_model_cached():
    try:
        # Compatibility shim: older pickles may reference _RemainderColsList which
        # does not exist in current sklearn. Provide a dummy to satisfy unpickling.
        if not hasattr(_ct, '_RemainderColsList'):
            class _RemainderColsList(list):
                pass
            _ct._RemainderColsList = _RemainderColsList
        return joblib.load('order_cancel_pipeline.pkl')
    except FileNotFoundError:
        st.error("Model file 'order_cancel_pipeline.pkl' not found. Please run train.py to generate it.")
        return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_meta_cached():
    try:
        with open('feature_meta.json','r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("feature_meta.json not found. Using minimal defaults.")
        return {}
    except Exception as e:
        st.warning(f"Failed to load feature metadata: {e}. Using minimal defaults.")
        return {}

model = load_model_cached()
meta = load_meta_cached() or {}

if model is None:
    st.stop()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 💰 Order Details")
    
    # Numeric inputs with better formatting
    order_amount = st.number_input("💵 Order Amount (LKR)", min_value=0.0, value=200.0, step=10.0, help="Total order value in Sri Lankan Rupees")
    number_of_items = st.number_input("📦 Number of Items", min_value=1, value=2, step=1, help="Total number of items in the order")
    distance_km = st.number_input("🚚 Distance (km)", min_value=0.0, value=5.0, step=0.5, help="Delivery distance in kilometers")
    customer_rating = st.number_input("⭐ Customer Rating (0-5)", min_value=0.0, max_value=5.0, value=4.0, step=0.1, help="Customer's average rating")
    
    st.markdown("### 📊 Customer History")
    previous_cancellations = st.number_input("❌ Previous Cancellations", min_value=0, value=0, step=1, help="Number of previous cancellations by this customer")
    delivery_duration_min = st.number_input("⏱️ Delivery Duration (min)", min_value=0.0, value=30.0, step=5.0, help="Expected delivery time in minutes")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 🏪 Restaurant & Location")
    
    # Categorical inputs with better formatting and explicit index
    restaurant_options = meta.get('restaurant', ['Other'])
    restaurant = st.selectbox("🍽️ Restaurant", restaurant_options, index=0, help="Select the restaurant")
    
    city_options = meta.get('city', ['Unknown'])
    city = st.selectbox("🏙️ City", city_options, index=0, help="Select the city")
    
    delivery_type_options = meta.get('delivery_type', ['Home Delivery', 'Pick-up'])
    delivery_type = st.selectbox("🚚 Delivery Type", delivery_type_options, index=0, help="Choose delivery method")
    
    st.markdown("### ⏰ Timing & Payment")
    
    order_time_options = meta.get('order_time', ['Morning','Afternoon','Evening','Night'])
    order_time = st.selectbox("🕐 Order Time", order_time_options, index=0, help="Time of day when order was placed")
    
    day_of_week_options = meta.get('day_of_week', ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    day_of_week = st.selectbox("📅 Day of Week", day_of_week_options, index=0, help="Day when order was placed")
    
    payment_method_options = meta.get('payment_method', ['Cash','Debit Card','Credit Card','Apple Pay','Google Pay'])
    payment_method = st.selectbox("💳 Payment Method", payment_method_options, index=0, help="Payment method used")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction button and results
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button("🔮 Predict Cancellation Probability", use_container_width=True)

if predict_button:
    # Prepare data for prediction
    Xnew = pd.DataFrame([{
        'order_amount': order_amount,
        'number_of_items': number_of_items,
        'distance_km': distance_km,
        'customer_rating': customer_rating,
        'previous_cancellations': previous_cancellations,
        'delivery_duration_min': delivery_duration_min,
        'restaurant': restaurant,
        'city': city,
        'delivery_type': delivery_type,
        'order_time': order_time,
        'day_of_week': day_of_week,
        'payment_method': payment_method
    }])
    
    # Get predictions
    prob = model.predict_proba(Xnew)[:,1][0]
    pred = model.predict(Xnew)[0]
    
    # Display results with beautiful styling
    st.markdown("---")
    
    # Main prediction result
    result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
    
    with result_col2:
        if pred == 1:
            st.markdown(f'''
            <div class="prediction-result">
                <h2>⚠️ High Cancellation Risk</h2>
                <h1 style="font-size: 3rem; margin: 1rem 0;">{prob:.1%}</h1>
                <p style="font-size: 1.2rem;">This order has a high probability of being cancelled</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="prediction-result success-result">
                <h2>✅ Low Cancellation Risk</h2>
                <h1 style="font-size: 3rem; margin: 1rem 0;">{prob:.1%}</h1>
                <p style="font-size: 1.2rem;">This order is likely to be completed successfully</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Detailed metrics and visualizations
    st.markdown("### 📊 Detailed Analysis")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Cancellation Probability", f"{prob:.1%}", delta=f"{prob-0.5:.1%}" if prob > 0.5 else f"{0.5-prob:.1%}")
    
    with metric_col2:
        risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
        st.metric("Risk Level", risk_level)
    
    with metric_col3:
        confidence = abs(prob - 0.5) * 2
        st.metric("Confidence", f"{confidence:.1%}")
    
    with metric_col4:
        recommendation = "Monitor closely" if prob > 0.7 else "Standard process" if prob > 0.3 else "Low priority"
        st.metric("Recommendation", recommendation)
    
    # Visualization
    viz_col1, viz_col2 = st.columns([1, 1])
    
    with viz_col1:
        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Cancellation Probability (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with viz_col2:
        # Risk factors analysis
        risk_factors = {
            'Order Amount': order_amount,
            'Distance': distance_km,
            'Delivery Time': delivery_duration_min,
            'Customer Rating': customer_rating,
            'Previous Cancellations': previous_cancellations
        }
        
        # Normalize values for visualization
        normalized_factors = {}
        for factor, value in risk_factors.items():
            if factor == 'Customer Rating':
                normalized_factors[factor] = (5 - value) / 5  # Higher rating = lower risk
            elif factor == 'Order Amount':
                normalized_factors[factor] = min(value / 1000, 1)  # Cap at 1000
            elif factor == 'Distance':
                normalized_factors[factor] = min(value / 20, 1)  # Cap at 20km
            elif factor == 'Delivery Time':
                normalized_factors[factor] = min(value / 60, 1)  # Cap at 60min
            else:
                normalized_factors[factor] = min(value / 5, 1)  # Cap at 5
        
        fig_bar = px.bar(
            x=list(normalized_factors.values()),
            y=list(normalized_factors.keys()),
            orientation='h',
            title="Risk Factor Analysis",
            color=list(normalized_factors.values()),
            color_continuous_scale="Reds"
        )
        fig_bar.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Recommendations section
    st.markdown("### 💡 Recommendations")
    
    recommendations = []
    if prob > 0.7:
        recommendations.extend([
            "🔔 Send proactive communication to customer",
            "⚡ Prioritize this order for faster processing",
            "📞 Consider calling customer to confirm details",
            "🎁 Offer incentives to reduce cancellation risk"
        ])
    elif prob > 0.3:
        recommendations.extend([
            "📱 Send order confirmation message",
            "⏰ Monitor delivery progress closely",
            "📋 Ensure accurate order details"
        ])
    else:
        recommendations.extend([
            "✅ Standard processing is sufficient",
            "📊 Include in regular monitoring",
            "🎯 Focus resources on higher-risk orders"
        ])
    
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    # Order summary
    st.markdown("### 📋 Order Summary")
    summary_data = {
        "Restaurant": restaurant,
        "City": city,
        "Order Amount": f"LKR {order_amount:,.0f}",
        "Items": f"{number_of_items} items",
        "Delivery Type": delivery_type,
        "Payment Method": payment_method,
        "Order Time": f"{order_time} ({day_of_week})",
        "Distance": f"{distance_km} km",
        "Expected Duration": f"{delivery_duration_min} min"
    }
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        for key, value in list(summary_data.items())[:5]:
            st.markdown(f"**{key}**: {value}")
    
    with summary_col2:
        for key, value in list(summary_data.items())[5:]:
            st.markdown(f"**{key}**: {value}")

