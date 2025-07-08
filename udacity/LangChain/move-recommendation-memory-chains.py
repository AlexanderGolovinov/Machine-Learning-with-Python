from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, CombinedMemory, ChatMessageHistory
from langchain.chains import ConversationChain
from typing import Any, Dict, Optional, Tuple

import requests
import os

model_name = "gpt-3.5-turbo"
temperature = 1.2
base_url = "https://openai.vocareum.com/v1"
api_key = os.environ.get("OPEN_AI_KEY")


def get_movie_plot(movie_name):
    headers = {
        'User-Agent': 'MoviePlotFetcher/1.0'
    }

    base_url = f"https://en.wikipedia.org/w/api.php"

    def is_movie_page(title):
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "categories|revisions",
            "rvprop": "content",
            "cllimit": "max"
        }

        response = requests.get(base_url, headers=headers, params=params)
        data = response.json()

        try:
            page = list(data["query"]["pages"].values())[0]

            # Check categories for Movie indication
            categories = [cat["title"] for cat in page.get("categories", [])]
            for category in categories:
                if "films" in category.lower():
                    return True

            # Check for infobox movie in the page content
            content = page["revisions"][0]["*"]
            if "{{Infobox film" in content:
                return True

        except Exception as e:
            pass

        return False

    def extract_plot_from_text(full_text):
        try:
            # Find the start of the Plot section
            plot_start = full_text.index("== Plot ==") + len("== Plot ==")

            # Find the start of the next section
            next_section_start = full_text.find("==", plot_start)

            # If no next section is found, use the end of the text
            if next_section_start == -1:
                next_section_start = len(full_text)

            # Extract the plot text and strip leading/trailing whitespace
            plot_text = full_text[plot_start:next_section_start].strip()

            # Return the extracted plot
            return plot_text

        except ValueError:
            # Return a message if the Plot section isn't found
            return "Plot section not found in the text."

    def extract_first_paragraph(full_text):
        # Find the first double newline
        end_of_first_paragraph = full_text.find("\n\n")

        # If found, slice the string to get the first paragraph
        if end_of_first_paragraph != -1:
            return full_text[:end_of_first_paragraph].strip()

        # If not found, return the whole text as it might be just one paragraph
        return full_text.strip()

    search_params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": movie_name,
        "utf8": 1,
        "srlimit": 5  # Top 5 search results
    }

    response = requests.get(base_url, headers=headers, params=search_params)
    data = response.json()

    # Go through top search results to find a movie page
    for search_result in data["query"]["search"]:
        title = search_result["title"]
        if is_movie_page(title):
            # Fetch plot for the movie page
            plot_params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            }

            plot_response = requests.get(base_url, headers=headers, params=plot_params)
            plot_data = plot_response.json()

            try:
                page = list(plot_data["query"]["pages"].values())[0]
                full_text = page.get("extract", "No text...")
                return f"""Overview:\n{extract_first_paragraph(full_text)}\nPlot:\n{extract_plot_from_text(full_text)}""".strip()
            except:
                return "Error fetching plot."

    return "Movie not found."


temperature = 0.0
llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name,
    temperature=temperature,
    max_tokens=1000)

movies = ["Barbie", "Oppenheimer", "The Notebook", "Dumb Money"]

personal_questions = ["Which movie genre you like the most?",
                      "What is your favorite color?",
                      "What is your favorite movie?",
                      "Pick one - dogs, cats or hamsters?",
                      "What is your favorite food?",
                      "What is your favorite drink?"]

# personal_answers = [ ]
# for question in personal_questions:
#    answer = input(question)
#    personal_answers.append(answer)

max_rating = 100

personal_answers = ['thriller', 'blue', 'inception', 'dogs', 'fish tacos', 'cold beer']
print(personal_answers)

history = ChatMessageHistory()
history.add_user_message(
    f"""You are AI that will recommend user a movie based on their answers to personal questions. Ask user {len(personal_questions)} questions""")
for i in range(len(personal_questions)):
    history.add_ai_message(personal_questions[i])
    history.add_user_message(personal_answers[i])

history.add_ai_message(
    """Now tell me a plot summary of a movie you're considering watching, and specify how you want me to respond to you with the movie rating""")
summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="recommendation_summary",
    input_key="input",
    buffer=f"The human answered {len(personal_questions)} personal questions). Use them to rate, from 1 to {max_rating}, how much they like a movie they describe to you.",
    return_messages=True)


# you could choose to store some of the q/a in memory as well, in addition to original questions
class MementoBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_ai_message(output_str)


conversational_memory = MementoBufferMemory(
    chat_memory=history,
    memory_key="questions_and_answers",
    input_key="input"
)

# Combined
memory = CombinedMemory(memories=[conversational_memory, summary_memory])
RECOMMENDER_TEMPLATE = """The following is a friendly conversation between a human and an AI Movie Recommender. 
                        The AI is follows human instructions and provides movie ratings for a human based on the movie plot
                        and human's persona derived from their answers to questions. 

Summary of Recommendations:
{recommendation_summary}
Personal Questions and Answers:
{questions_and_answers}
Human: {input}
AI:"""
PROMPT = PromptTemplate(
    input_variables=["recommendation_summary", "input", "questions_and_answers"],
    template=RECOMMENDER_TEMPLATE
)
recommender = ConversationChain(llm=llm, verbose=True, memory=memory, prompt=PROMPT)

for movie in movies:
    print("Movie: " + movie)
    movie_plot = get_movie_plot(movie)
    # print(f"Plot: {movie_plot}")

    plot_rating_instructions = f"""
         =====================================
        === START MOVIE PLOT SUMMARY FOR {movie} ===
        {movie_plot}
        === END MOVIE PLOT SUMMARY ===
        =====================================

        RATING INSTRUCTIONS THAT MUST BE STRICTLY FOLLOWED:
        AI will provide a highly personalized rating based only on the movie plot summary human provided 
        and human answers to questions included with the context. 
        AI should be very sensible to human personal preferences captured in the answers to personal questions, 
        and should not be influenced by anything else.
        AI will also build a persona for human based on human answers to questions, and use this persona to rate the movie.
        OUTPUT FORMAT:
        First, include that persona you came up with in the explanation for the rating. Describe the persona in a few sentences.
        Explain how human preferences captured in the answers to personal questions influenced creation of this persona.
        In addition, consider other ratings for this human that you might have as they might give you more information about human's preferences.
        Your goal is to provide a rating that is as close as possible to the rating human would give to this movie.
        Remember that human has very limited time and wants to see something they will like, so your rating should be as accurate as possible.
        Rating will range from 1 to {max_rating}, with {max_rating} meaning human will love it, and 1 meaning human will hate it. 
        You will include a logical explanation for your rating based on human persona you've build and human responses to questions.
        YOUR REVIEW MUST END WITH TEXT: "RATING FOR MOVIE {movie} is " FOLLOWED BY THE RATING.
        FOLLOW THE INSTRUCTIONS STRICTLY, OTHERWISE HUMAN WILL NOT BE ABLE TO UNDERSTAND YOUR REVIEW.
    """
    prediction = recommender.predict(input=plot_rating_instructions)
    print(prediction)

final_recommendation = """Now that AI has rated all the movies, AI will recommend human the one that human will like the most. 
                            AI will respond with movie recommendation, and short explanation for why human will like it over all other movies. 
                            AI will not include any ratings in your explanation, only the reasons why human will like it the most.
                            However, the movie you will pick must be one of the movies you rated the highest.
                            For example, if you rated one movie 65, and the other 60, you will recommend the movie with rating 65 because rating 65 
                            is greate than rating of 60 ."""
prediction = recommender.predict(input=final_recommendation)
print(prediction)