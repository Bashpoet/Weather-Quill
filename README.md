# Weather's Quill Meteorological Dramaturgy: Real-Time Data Meets Theatrical AI Prose

**Project Title**  
*Meteorological Dramaturgy: Real-Time Data Meets Theatrical AI Prose*

---

# Meteorological Dramaturgy

Ever wondered what it would be like if your local forecast were spun into a nightly Shakespearean soliloquy or an operatic ode? **Meteorological Dramaturgy** blends real-time weather data from the OpenWeatherMap API with an LLM’s capacity to weave lyrical, dramatic narratives. Each time the script runs, the atmospheric data you receive—humidity, temperature, wind speed, pressure—gets cast as living, breathing characters on a cosmic stage, resulting in a one-of-a-kind textual performance tailored to the exact weather of the moment.

## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Prerequisites](#prerequisites)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Customization](#customization)  
7. [Possible Future Developments](#possible-future-developments)  
8. [License](#license)

---

## Overview

**Meteorological Dramaturgy** fetches current weather data—like temperature, wind, humidity—and hands it off to a Large Language Model (LLM) via a system prompt. The LLM, instructed to personify and dramatize the atmospheric variables, composes a brand-new piece of prose every time you run the script.

You can think of it as:

1. **Data Collection:** We gather fresh meteorological data from an API.  
2. **Theatrical Writing:** An AI (GPT-based or otherwise) generates a short, imaginative forecast using dramatic metaphors (e.g., humidity as a “sultry contralto,” pressure gradients as “conspirators,” clouds as “celestial actors,” etc.).  
3. **Continuous Novelty:** Each call to the script yields different text, unique to the weather data at that moment in time.

---

## Features

- **Real-Time Weather**: Gets live conditions via [OpenWeatherMap](https://openweathermap.org).  
- **AI-Generated Prose**: Feeds data to an LLM, returning comedic, poetic, or downright operatic narratives.  
- **Easily Extensible**: Swap in different data sources or incorporate more meteorological parameters (e.g., sunrise, sunset, UV index).  
- **Beginner-Friendly**: The code is written in Python and annotated with docstrings, making it accessible for those new to APIs or AI.  

---

## Prerequisites

1. **Python 3.7+**  
2. **Python Libraries**  
   - `requests` for HTTP requests  
   - `openai` (or your chosen LLM library) for AI interaction  
3. **OpenWeatherMap API Key**  
   - Sign up at [OpenWeatherMap](https://openweathermap.org/) for a free API key  
4. **OpenAI API Key** (or other LLM provider credentials)  
   - Sign up at [OpenAI](https://openai.com/) or another provider, then grab your API key  

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/<your-username>/meteorological-dramaturgy.git
   cd meteorological-dramaturgy
   ```
2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   If you haven’t yet generated a `requirements.txt`, you can do:
   ```bash
   pip install requests openai
   ```
3. **Set Up Environment Variables** (Recommended)
   - Create a `.env` file containing:
     ```
     OPENWEATHER_API_KEY=your_openweathermap_key
     OPENAI_API_KEY=your_openai_key
     ```
   - Or just manually replace placeholders in the script with your keys.

---

## Usage

1. **Run the main script**:
   ```bash
   python theatrical_weather.py
   ```
2. **Enter your city** when prompted:
   ```
   Enter a city name for the grand meteorological opera: Berlin
   ```
3. **Marvel** at the bespoke, lyrical forecast that appears in your console.

Every time you run the script, it fetches fresh data and the LLM composes a new “act” that might personify the wind as a “gentle but cunning orchestrator” or paint the humidity as a “velvet drape lingering on the horizon.”  

---

## Customization

- **Prompt Engineering**  
  Open `theatrical_weather.py` (or whichever file you named it) and edit the `SYSTEM_PROMPT` string to tweak the style or tone.  
- **Extra Data Fields**  
  In `process_weather_data`, pull additional parameters (like sunrise, sunset, feels-like temperature) and feed them into the user prompt.  
- **Temperature Control**  
  Adjust the `temperature` in the LLM call—higher values yield more imaginative (and sometimes whimsical) prose, lower values yield more factual, reserved descriptions.

---

## Possible Future Developments

- **Forecast Opera**: Extend the code to retrieve 24-hour or 5-day forecasts, generating multi-act dramas with different plot twists for morning, afternoon, and night.  
- **User-Selectable Themes**: Let the user pick whether the style should be Shakespearian, Romantic, Sci-Fi, or even Film Noir.  
- **Web Interface**: Build a simple Flask or Streamlit app so visitors can input a location and get a spontaneously generated reading.  
- **Analytics**: Track how the LLM references changes in weather data across runs and store them for further creative or educational analysis.

---

## License

This project is released under the [MIT License](LICENSE). You’re free to modify, distribute, or build upon it. Have fun turning raw meteorological data into ephemeral sky poetry!

---

### Enjoy the performance

Whether you’re curious about AI’s capabilities, teaching yourself to consume a public API, or just love the idea of reading a *tempestuous meteorological monologue* each morning, **Meteorological Dramaturgy** offers a playful fusion of creativity and code. Give it a spin, add your own twists, and help expand the grand stage that is our atmosphere. Have fun!
