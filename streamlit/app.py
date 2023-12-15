# Streamlit dependencies
import streamlit as st
import joblib,os

st.set_page_config(page_title="Tweet Classifer", page_icon=":cloud:", layout="wide")

bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url('https://images.unsplash.com/photo-1649389966541-12af5cb7efa0?q=80&w=1770&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
background-size: 100% 100%;
background-repeat: no-repeat;
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}

[data-testid="stToolbar"] {
right: 2rem;
}

[data-testid="stSidebar"] {
background-image: url('https://images.unsplash.com/photo-1616945459385-6d4248514cc7?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
background-size: cover;
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
	st.title(":red[Block]IT")
	st.subheader("Hate Speech or Offensive Language Detection")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Prediction", "Credits"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("""Welcome to BlockIT, your powerful tool for promoting positive online interactions!
			  In a world where communication happens at the speed of a tweet, it's essential to foster a digital environment that is respectful and inclusive.
			  Our app employs cutting-edge technology to analyze and classify tweets, helping users identify and understand the tone of online content.

**Key Features:**

1. **Hate Speech Detection:**
   Uncover harmful language and discriminatory content. Our app utilizes advanced algorithms to identify tweets containing hate speech, promoting a safer and more inclusive digital space.

2. **Offensive Language Recognition:**
   Stay ahead of negativity by detecting offensive language in tweets. We strive to create a platform where users can express themselves without fear of encountering offensive content.

3. **Neutral Content Classification:**
   Not all tweets are harmful. Our app also distinguishes and categorizes tweets as neutral, ensuring that users can focus on engaging with content that contributes positively to the online discourse.

**How It Works:**

- **Real-time Analysis:**
  Get instant feedback on the tweets you encounter. Our app works seamlessly to classify content as it appears in your feed, providing real-time insights into the nature of the tweets.

- **User-Friendly Interface:**
  BlockIT is designed with simplicity in mind. With an intuitive interface, users can effortlessly navigate through the app, making it easy to understand and utilize its features.

- **Personalized Settings:**
  Tailor the app to your preferences. Adjust settings to match your comfort level, allowing you to strike the right balance between free expression and a respectful online environment.

**Why BlockIT?**

- **Community Well-being:**
  By using BlockIT, you contribute to a digital community that values respect and understanding. Together, we can make social media a space where everyone feels safe to express themselves.

- **Empowerment Through Awareness:**
  Knowledge is power. BlockIT empowers users by providing insights into the content they consume, fostering a sense of awareness and responsibility in online interactions.

Join us in creating a positive online experience. Download BlockIT now and take control of your digital environment! Together, let's build a more inclusive and respectful online community.""")

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
		label_mapping = {'Hate Speech': 0, 'Offensive Language': 1, 'Neither Hate Speech nor Offensive Language': 2}

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
		st.text("Ajirioghene Oguh")
		st.subheader('Images')
		st.text("""Background Image\t\tISTOCK / GETTY IMAGES PLUS \n\nSidebar Image\t\t\tJason Leung (unsplash.com)""")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
