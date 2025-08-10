import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CompuLaw AI - Legal Intelligence Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# LOAD MODEL AND DATA
# =====================================================

@st.cache_data
def load_model_and_data():
    """Load the trained model and clean data"""
    try:
        with open('compulaw_ai_model.pkl', 'rb') as f:
            model_data = pickle.load(f)

        df_clean = pd.read_csv('cleaned_supreme_court_data.csv')

        # Clean up any remaining NaN values in categorical columns
        categorical_cols = ['offence', 'appeal_district', 'trial_district', 'sentence']
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna('Unknown')

        return model_data, df_clean

    except FileNotFoundError as e:
        st.error(f"Required files not found: {e}")
        return None, None


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_case_outcome(case_details, model_data):
    """Predict outcome for a specific case with detailed analysis"""

    model = model_data['model']
    encoders = model_data['encoders']
    feature_columns = model_data['feature_columns']

    # Create a single row dataframe
    case_df = pd.DataFrame([case_details])

    # Encode categorical features
    categorical_features = ['offence', 'appeal_district', 'trial_district', 'sentence']

    for feature in categorical_features:
        if feature in case_df.columns:
            case_df[feature] = case_df[feature].fillna('Unknown')
            try:
                case_df[f'{feature}_encoded'] = encoders[feature].transform(case_df[feature])
            except ValueError:
                # Handle unseen categories
                case_df[f'{feature}_encoded'] = 0

    # Select features and predict
    case_features = case_df[feature_columns].fillna(0)
    prediction = model.predict(case_features)[0]
    probability = model.predict_proba(case_features)[0]

    # Get probabilities for both outcomes
    prob_dismissed = probability[0] if model.classes_[0] == 'Dismissed' else probability[1]
    prob_granted = probability[1] if model.classes_[1] == 'Granted' else probability[0]

    # Get feature importance for this specific case
    feature_importance = model.feature_importances_

    return {
        'prediction': prediction,
        'prob_dismissed': prob_dismissed,
        'prob_granted': prob_granted,
        'confidence': max(probability),
        'feature_importance': feature_importance
    }


# =====================================================
# MAIN APP
# =====================================================

def main():
    # Load data
    model_data, df_clean = load_model_and_data()

    if model_data is None or df_clean is None:
        st.error("⚠️ Model files not found. Please run the training steps first.")
        st.markdown("""
        **Required files:**
        1. `compulaw_ai_model.pkl` (from Step 3)
        2. `cleaned_supreme_court_data.csv` (from Step 1)

        Please run the previous notebook cells to generate these files.
        """)
        return

    # App header
    st.title("⚖️ CompuLaw AI - Legal Intelligence Platform")
    st.markdown("### *Democratizing Legal Intelligence for Nigerian Lawyers*")

    st.markdown("""
    ---
    **CompuLaw AI** uses artificial intelligence to analyze Nigerian Supreme Court appeal cases 
    and provide intelligent insights to help lawyers make better decisions.
    """)

    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["🎯 Case Outcome Predictor", "📊 Legal Intelligence Dashboard", "🔍 Why You Lost Analyzer", "📈 Market Insights"]
    )

    if page == "🎯 Case Outcome Predictor":
        show_case_predictor(model_data, df_clean)
    elif page == "📊 Legal Intelligence Dashboard":
        show_dashboard(df_clean)
    elif page == "🔍 Why You Lost Analyzer":
        show_why_you_lost(model_data, df_clean)
    elif page == "📈 Market Insights":
        show_market_insights(df_clean)


# =====================================================
# CASE OUTCOME PREDICTOR
# =====================================================

def show_case_predictor(model_data, df_clean):
    st.header("🎯 Case Outcome Predictor")
    st.markdown("**Predict the likelihood of your appeal being granted or dismissed**")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Enter Case Details:")

        # Case input form
        with st.form("case_prediction_form"):
            # Basic case information
            offense_options = sorted(df_clean['offence'].unique())
            offense = st.selectbox("Offense Type:", offense_options)

            region_options = sorted(df_clean['appeal_district'].unique())
            appeal_region = st.selectbox("Appeal District:", region_options)

            trial_region_options = sorted(df_clean['trial_district'].dropna().unique())
            trial_region = st.selectbox("Trial District:", trial_region_options)

            sentence_options = sorted(df_clean['sentence'].dropna().unique())
            sentence = st.selectbox("Original Sentence:", sentence_options)

            # People involved
            st.markdown("**People Involved:**")
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                no_complainant = st.number_input("Complainants:", min_value=0, max_value=20, value=1)
                no_male_complainant = st.number_input("Male Complainants:", min_value=0, max_value=20, value=1)
                no_female_complainant = st.number_input("Female Complainants:", min_value=0, max_value=20, value=0)

            with col_b:
                no_appealant = st.number_input("Appellants:", min_value=0, max_value=20, value=1)
                no_male_appealant = st.number_input("Male Appellants:", min_value=0, max_value=20, value=1)
                no_female_appealant = st.number_input("Female Appellants:", min_value=0, max_value=20, value=0)

            with col_c:
                no_public_witness = st.number_input("Public Witnesses:", min_value=0, max_value=20, value=2)
                no_eye_witness = st.number_input("Eye Witnesses:", min_value=0, max_value=20, value=1)
                no_defense_witness = st.number_input("Defense Witnesses:", min_value=0, max_value=20, value=1)

            predict_button = st.form_submit_button("🔮 Predict Outcome", type="primary")

        if predict_button:
            # Prepare case details
            case_details = {
                'offence': offense,
                'appeal_district': appeal_region,
                'trial_district': trial_region,
                'sentence': sentence,
                'no_complainant': no_complainant,
                'no_male_complainant': no_male_complainant,
                'no_female_complainant': no_female_complainant,
                'no_appealant': no_appealant,
                'no_male_appealant': no_male_appealant,
                'no_female_appealant': no_female_appealant,
                'no_public_witness': no_public_witness,
                'no_eye_witness': no_eye_witness,
                'no_defense_witness': no_defense_witness
            }

            # Make prediction
            result = predict_case_outcome(case_details, model_data)

            with col2:
                st.subheader("🎯 Prediction Results")

                # Main prediction
                if result['prediction'] == 'Granted':
                    st.success(f"**LIKELY TO BE GRANTED** ✅")
                else:
                    st.error(f"**LIKELY TO BE DISMISSED** ❌")

                # Probability breakdown
                st.markdown("**Probability Breakdown:**")

                # Create probability chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Dismissed', 'Granted'],
                        y=[result['prob_dismissed'] * 100, result['prob_granted'] * 100],
                        marker_color=['#ff4444', '#44ff44'],
                        text=[f"{result['prob_dismissed'] * 100:.1f}%", f"{result['prob_granted'] * 100:.1f}%"],
                        textposition='auto',
                    )
                ])

                fig.update_layout(
                    title="Outcome Probabilities",
                    yaxis_title="Probability (%)",
                    height=300,
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

                # Confidence and recommendations
                st.metric("Model Confidence", f"{result['confidence'] * 100:.1f}%")

                # Historical context
                similar_cases = df_clean[
                    (df_clean['offence'] == offense) &
                    (df_clean['appeal_district'] == appeal_region)
                    ]

                if len(similar_cases) > 0:
                    historical_success = (similar_cases['scn_decision'] == 'Granted').mean()
                    st.info(
                        f"📊 Historical data: {len(similar_cases)} similar cases with {historical_success:.1%} success rate")


# =====================================================
# LEGAL INTELLIGENCE DASHBOARD
# =====================================================

def show_dashboard(df_clean):
    st.header("📊 Legal Intelligence Dashboard")
    st.markdown("**Comprehensive insights from Nigerian Supreme Court appeals**")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    total_cases = len(df_clean)
    granted_cases = (df_clean['scn_decision'] == 'Granted').sum()
    success_rate = granted_cases / total_cases

    with col1:
        st.metric("Total Cases Analyzed", f"{total_cases:,}")
    with col2:
        st.metric("Cases Granted", f"{granted_cases:,}")
    with col3:
        st.metric("Overall Success Rate", f"{success_rate:.1%}")
    with col4:
        st.metric("Offense Types", df_clean['offence'].nunique())

    # Charts
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Success Rates by Offense Type")

        # Calculate success rates
        offense_success = df_clean.groupby('offence').agg({
            'scn_decision': ['count', lambda x: (x == 'Granted').sum()]
        })
        offense_success.columns = ['total', 'granted']
        offense_success['success_rate'] = offense_success['granted'] / offense_success['total']
        offense_success = offense_success[offense_success['total'] >= 20].sort_values('success_rate', ascending=True)

        fig = px.bar(
            x=offense_success['success_rate'] * 100,
            y=offense_success.index,
            orientation='h',
            title="Success Rate by Offense (20+ cases)",
            labels={'x': 'Success Rate (%)', 'y': 'Offense Type'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Regional Performance")

        regional_success = df_clean.groupby('appeal_district').agg({
            'scn_decision': ['count', lambda x: (x == 'Granted').sum()]
        })
        regional_success.columns = ['total', 'granted']
        regional_success['success_rate'] = regional_success['granted'] / regional_success['total']

        fig = px.pie(
            values=regional_success['total'],
            names=regional_success.index,
            title="Case Distribution by Region"
        )
        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# WHY YOU LOST ANALYZER
# =====================================================

def show_why_you_lost(model_data, df_clean):
    st.header("🔍 Why You Lost Analyzer")
    st.markdown("**Analyze what factors contributed to case dismissal and get improvement suggestions**")

    st.markdown("""
    This feature helps lawyers understand why cases are dismissed and provides actionable 
    recommendations for strengthening future appeals.
    """)

    # Sample dismissed case analysis
    dismissed_cases = df_clean[df_clean['scn_decision'] == 'Dismissed'].sample(5)

    st.subheader("Sample Case Analysis")

    for idx, case in dismissed_cases.iterrows():
        with st.expander(f"Case {idx}: {case['offence']} in {case['appeal_district']}"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Case Details:**")
                st.write(f"- Offense: {case['offence']}")
                st.write(f"- Region: {case['appeal_district']}")
                st.write(f"- Original Sentence: {case['sentence']}")
                st.write(f"- Eye Witnesses: {case['no_eye_witness']}")
                st.write(f"- Defense Witnesses: {case['no_defense_witness']}")

            with col2:
                st.markdown("**Possible Improvement Factors:**")

                # Get similar successful cases
                similar_granted = df_clean[
                    (df_clean['offence'] == case['offence']) &
                    (df_clean['scn_decision'] == 'Granted')
                    ]

                if len(similar_granted) > 0:
                    avg_defense_witnesses = similar_granted['no_defense_witness'].mean()
                    avg_eye_witnesses = similar_granted['no_eye_witness'].mean()

                    if case['no_defense_witness'] < avg_defense_witnesses:
                        st.write(
                            f"✅ Consider more defense witnesses (avg in successful cases: {avg_defense_witnesses:.1f})")

                    if case['no_eye_witness'] > avg_eye_witnesses:
                        st.write(
                            f"⚠️ High number of eye witnesses may hurt case (avg in successful: {avg_eye_witnesses:.1f})")

                # Regional advice
                best_region = df_clean.groupby('appeal_district')['scn_decision'].apply(
                    lambda x: (x == 'Granted').mean()
                ).idxmax()

                if case['appeal_district'] != best_region:
                    st.write(f"📍 Consider jurisdiction: {best_region} has higher success rates")


# =====================================================
# MARKET INSIGHTS
# =====================================================

def show_market_insights(df_clean):
    st.header("📈 Market Insights")
    st.markdown("**Strategic intelligence for law firms and legal practitioners**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Most Winnable Case Types")

        offense_success = df_clean.groupby('offence').agg({
            'scn_decision': ['count', lambda x: (x == 'Granted').sum()]
        })
        offense_success.columns = ['total', 'granted']
        offense_success['success_rate'] = offense_success['granted'] / offense_success['total']

        winnable_cases = offense_success[offense_success['total'] >= 20].sort_values('success_rate',
                                                                                     ascending=False).head(5)

        for offense, data in winnable_cases.iterrows():
            st.write(f"**{offense}**: {data['success_rate']:.1%} success rate ({data['total']} cases)")

    with col2:
        st.subheader("🎯 Strategic Recommendations")

        st.markdown("""
        **For Law Firms:**
        - Focus on trespassing and property dispute cases
        - Consider regional jurisdiction advantages
        - Build strong defense witness strategies

        **For Individual Lawyers:**
        - Specialize in high-success offense types
        - Develop expertise in North-East region appeals
        - Focus on civil rather than criminal appeals

        **For Legal Education:**
        - Train on witness management strategies 
        - Study regional court preferences
        - Develop case strengthening methodologies
        """)


# =====================================================
# FOOTER
# =====================================================

def show_footer():
    st.markdown("---")
    st.markdown("""
    **CompuLaw AI** - Built with ❤️ for Nigerian legal practitioners  
    *Powered by AI • Grounded in Data • Focused on Justice*  

    📧 Contact: [Your Email] | 🌐 Website: [Your Website]
    """)


# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":
    main()
    show_footer()