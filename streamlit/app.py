# Streamlit dependencies
import streamlit as st
import joblib,os

st.set_page_config(page_title="Tweet Classifer", page_icon=":cloud:", layout="wide")

bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url('https://images.unsplash.com/photo-1605778336817-121ba9819b96?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1282&q=80');
background-size: cover;
background-position: top left;
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}

[data-testid="stToolbar"] {
right: 2rem;
}

[data-testid="stSidebar"] {
background-image: url('https://images.unsplash.com/photo-1610270197941-925ce9015c40?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1074&q=80');
background-size: center;
background-position: center;
}
</style>
"""
st.markdown(bg_img, unsafe_allow_html=True)

# Vectorizer
vectorizer = open("../src/dumps/cv.joblib","rb")
tweet_cv = joblib.load(vectorizer) # loading your vectorizer from the pkl file

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title(":green[SynapseAI] Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Prediction", "Credits"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("""Welcome to SynapseAI, a cutting-edge AI company revolutionizing industries worldwide.
		Our team of experts leverages advanced algorithms, machine learning, and natural language processing
		to deliver innovative solutions. From personalized virtual assistants to data analytics and automation,
		we empower businesses to thrive in the digital era. Join us on this transformative journey.
		Our ML model for sentiment analysis of climate change-related tweets combines the power of natural
		language processing and machine learning techniques to provide a comprehensive understanding of public
		sentiments. By leveraging this model, we can extract meaningful insights from the vast pool of Twitter
		data, enabling a data-driven approach towards addressing climate change and fostering informed decision-making.""")

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_input("Tweet", value="Enter your text here")

		# Load models from .pkl files
		lr = joblib.load(open(os.path.join("../src/dumps/lr_pipeline.joblib"),"rb"))
		mnb = joblib.load(open(os.path.join("../src/dumps/multiNB_pipeline.joblib"),"rb"))		

		model_list = [lr, mnb]

		model = st.selectbox('Select Model', options=model_list)

		# Create a dictionary to map prediction labels to human-readable categories
		label_mapping = {'Hate Speech': 0, 'Offensive': 1, 'Neither hate speech nor Offensive': 2}

		def get_key_from_value(dictionary, value):
			for key, val in dictionary.items():
				if val == value:
					return key
			return None

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			prediction = model.predict(vect_text)
			# When model has successfully run, will print prediction
			result = get_key_from_value(label_mapping, prediction)
			# more human interpretable.
			st.success("Tweet Categorized as: {}".format(result))

	# Building the Credits page
	if selection == "Credits":
		st.info('Credits')
		st.subheader('Author:')
		st.text("""Ajirioghene Oguh""")
		st.subheader('Images')
		st.text("""Background Image\t\tChristian Lue (unsplash.com)\n\nSidebar Image\t\t\tJohn Cameron (unsplash.com)""")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
