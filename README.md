# Weather-Quill  
**Meteorological Dramaturgy: Real-Time Data Meets Theatrical AI Prose**  

Ever wondered what it would be like if your local forecast were spun into a nightly Shakespearean soliloquy or an operatic ode? **Weather-Quill** blends real-time weather data from the OpenWeatherMap API with an LLM’s capacity to weave lyrical, dramatic narratives. Each time the script runs, the atmospheric data you receive—humidity, temperature, wind speed, pressure—gets cast as living, breathing characters on a cosmic stage, resulting in a one-of-a-kind textual performance tailored to the exact weather of the moment.

---

## Table of Contents  
1. [Overview](#overview)  
2. [Features](#features)  
3. [Prerequisites](#prerequisites)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Code Examples](#code-examples)  
7. [Customization](#customization)  
8. [Possible Future Developments](#possible-future-developments)  
9. [License](#license)  

---

## Overview
**Weather-Quill** fetches current weather data—like temperature, wind, humidity—and hands it off to a Large Language Model (LLM) via a system prompt. The LLM, instructed to personify and dramatize the atmospheric variables, composes a brand-new piece of prose every time you run the script.

You can think of it as:
1. **Data Collection:** Fresh meteorological data gets pulled from [OpenWeatherMap](https://openweathermap.org).  
2. **Theatrical Writing:** An AI (GPT-based or otherwise) generates short, imaginative forecasts teeming with metaphors—humidity might be a “sultry contralto,” while pressure gradients lurk like “conspirators.”  
3. **Continuous Novelty:** Each run produces text unique to that moment’s weather conditions, so you’re always reading a new “scene” of the sky’s drama.

---

## Features
- **Real-Time Weather**: Retrieves live conditions via [OpenWeatherMap](https://openweathermap.org).  
- **AI-Generated Prose**: Feeds data to an LLM, returning poetic, comedic, or downright operatic narratives.  
- **Beginner-Friendly**: Written in Python, with docstrings and comments to clarify how data flows from the API to the LLM.  
- **Easily Extensible**: Swap in different data sources or add more meteorological parameters (e.g., sunrise, sunset, UV index) for richer storytelling.

---

## Prerequisites
1. **Python 3.7+**  
   - Make sure you’ve installed Python version 3.7 or above.
2. **Python Libraries**  
   - `requests` for HTTP requests  
   - `openai` (or a similar LLM library) for AI interactions  
3. **OpenWeatherMap API Key**  
   - Sign up for a free API key at [OpenWeatherMap](https://openweathermap.org/)  
4. **OpenAI API Key** (or other LLM provider credentials)  
   - Sign up at [OpenAI](https://openai.com/) or with another provider, then grab your API key.  

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/<your-username>/weather-quill.git
   cd weather-quill
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   If you haven’t generated a `requirements.txt` yet, you can simply run:
   ```bash
   pip install requests openai
   ```
   and then:
   ```bash
   pip freeze > requirements.txt
   ```

3. **Set Up Environment Variables** (Recommended)  
   - Create a `.env` file at the root of the project with:
     ```
     OPENWEATHER_API_KEY=your_openweathermap_key
     OPENAI_API_KEY=your_openai_key
     ```
   - If you prefer, you can manually replace placeholders in the script with your keys, but storing them in `.env` is more secure and convenient.

4. **Confirm Your Environment**  
   - Run `python --version` to check you’re on Python 3.7+.
   - Create a virtual environment (optional but recommended):
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Linux/Mac
     .\venv\Scripts\activate   # On Windows
     ```

---

## Usage

1. **Run the Main Script**  
   From the project directory, type:
   ```bash
   python theatrical_weather.py
   ```
   (Or whatever you’ve named your script—e.g., `weather_quill.py`.)

2. **Enter Your City**  
   ```bash
   Enter a city name for the grand meteorological opera: Berlin
   ```
3. **Enjoy the Show**  
   The script fetches fresh weather data and the LLM composes a new “act,” weaving your local conditions into a theatrical monologue. You might read about a “gentle but cunning orchestrator” of a breeze, or a “velvet drape of humidity” at 82%.

---

## Code Examples

Below is a simplified snippet illustrating the core flow. For a full version, see the main file in the repo:

```python
import os
import requests
import openai

# Suppose your environment variables are in .env, loaded by dotenv
# from dotenv import load_dotenv
# load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are a dramaturge of the atmosphere, weaving weather data into lyrical, theatrical narratives.
Keep your commentary scientifically grounded but delivered in lively, poetically-charged prose.
"""

def fetch_weather(city):
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def build_prompt(weather_data):
    city_name = weather_data["name"]
    description = weather_data["weather"][0]["description"]
    temp = weather_data["main"]["temp"]
    humidity = weather_data["main"]["humidity"]

    return (
        f"City: {city_name}\n"
        f"Description: {description}\n"
        f"Temperature: {temp}°C\n"
        f"Humidity: {humidity}%\n"
        "Dramatize these details in a short theatrical commentary."
    )

def call_llm(system_prompt, user_prompt):
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response["choices"][0]["message"]["content"]

def main():
    city = input("Enter a city name for the grand meteorological opera: ")
    weather_data = fetch_weather(city)
    user_prompt = build_prompt(weather_data)
    print("\n--- Theatrical Weather Report ---\n")
    print(call_llm(SYSTEM_PROMPT, user_prompt))
    print("\n-------------------------------\n")

if __name__ == "__main__":
    main()
```

---

## Customization

1. **Prompt Engineering**  
   - Open your script and customize the `SYSTEM_PROMPT` text to change the writing style. You might go for a comedic, philosophical, or even noir-inspired tone.

2. **Extra Data Fields**  
   - In your `fetch_weather` or `build_prompt` functions, extract additional parameters like wind speed, sunrise, or cloud coverage. Mention them in the prompt so the AI can include them in the narrative.

3. **Temperature Control**  
   - Inside `call_llm`, adjusting `temperature=0.7` (or higher/lower) influences creativity. A higher number typically results in more flamboyant prose; a lower one yields more factual, sober writing.

4. **Error Handling**  
   - You can wrap your API calls and LLM requests in `try-except` blocks to manage timeouts or invalid city names gracefully.

---

## Possible Future Developments

- **Forecast Opera**: Retrieve multi-day forecasts to craft multi-act dramas—morning, afternoon, and night each get their own mini-scene.  
- **User-Selectable Themes**: Let users pick whether the style should be Shakespearean, Sci-Fi, Film Noir, or any other flavor before retrieving data.  
- **Analytics**: Track how the LLM references changes in weather data over time, potentially storing it for creative or scientific comparisons.  
- **Web Interface**: Build a simple Flask or Streamlit app so visitors can input a location and receive spontaneously generated readings and theatrical flair.  
- **Internationalization**: Offer multi-language support by adjusting the LLM prompt to produce commentary in different languages.

---

## License
This project is released under the [MIT License](LICENSE). You’re free to modify, distribute, or build upon it. Have fun turning raw meteorological data into ephemeral sky poetry!

---

### Enjoy the Performance
Whether you’re curious about AI’s creative range, eager to learn how to connect a public API to a Python script, or simply enchanted by the thought of a **tempestuous meteorological monologue** each morning, **Weather-Quill** offers a playful fusion of creativity and code. Give it a spin, experiment with custom styles, or expand the stage for an even grander atmospheric opera. Let the forecast be your muse!
