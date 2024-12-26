import os
import requests
import openai
import datetime

# If you're using environment variables, ensure you install python-dotenv and uncomment these lines:
# from dotenv import load_dotenv
# load_dotenv()

# --------------------------- Configuration Section --------------------------- #

# Replace these placeholders with either:
# 1) Actual keys (not recommended for production), or
# 2) Environment variables via os.getenv("VARIABLE_NAME").
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHERMAP_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# SYSTEM_PROMPT sets the “high-level direction” for how the LLM responds.
SYSTEM_PROMPT = """
You are a dramaturge of the atmosphere—a meteorological muse who weaves weather data into an immersive,
theatrical narrative. You adopt a playful, eloquent tone. Clouds, winds, humidity, and temperature are characters
on a grand stage. Pressure gradients conspire in the wings, humidity flirts at center stage, and the wind
acts as a gentle prompter or cunning director. Keep your commentary scientifically grounded but delivered in
lively, lyrical prose. Reference the data you receive as if unveiling a hidden script that reveals the drama overhead.
"""


# --------------------------- Weather Data Retrieval -------------------------- #

def fetch_weather_data(city_name: str, api_key: str) -> dict:
    """
    Makes an HTTP GET request to OpenWeatherMap, retrieving current weather data
    for a specified city. Returns the raw JSON data as a dictionary, or None if
    an error occurs.

    :param city_name: Name of the city (e.g., "Berlin" or "Tokyo").
    :param api_key: Your OpenWeatherMap API key.
    :return: Dictionary containing the raw JSON response, or None on failure.
    """
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_name,
        "appid": api_key,
        "units": "metric"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: Unable to retrieve data from OpenWeatherMap. Details: {e}")
        return None


# --------------------------- Data Processing --------------------------------- #

def process_weather_data(raw_data: dict) -> dict:
    """
    Extracts key fields from the raw JSON data and organizes them
    into a concise dictionary for use by the LLM prompt.

    :param raw_data: Dictionary containing raw JSON weather data from OpenWeatherMap.
    :return: Dictionary of processed weather information, or None if parsing fails.
    """
    if not raw_data:
        return None

    # Basic attributes
    city_name = raw_data.get("name", "Unknown City")

    weather_array = raw_data.get("weather", [{}])
    main_weather = weather_array[0].get("description", "No description")

    main_data = raw_data.get("main", {})
    temp = main_data.get("temp", 0.0)
    feels_like = main_data.get("feels_like", 0.0)
    humidity = main_data.get("humidity", 0)
    pressure = main_data.get("pressure", 0)

    wind_data = raw_data.get("wind", {})
    wind_speed = wind_data.get("speed", 0.0)  # meters/sec
    wind_deg = wind_data.get("deg", 0)

    cloud_data = raw_data.get("clouds", {})
    cloudiness = cloud_data.get("all", 0)

    # Optional extras
    sys_data = raw_data.get("sys", {})
    sunrise_unix = sys_data.get("sunrise")
    sunset_unix = sys_data.get("sunset")

    # Convert sunrise/sunset to human-readable times if present
    sunrise_time = (
        datetime.datetime.fromtimestamp(sunrise_unix).strftime('%H:%M:%S')
        if sunrise_unix
        else "N/A"
    )
    sunset_time = (
        datetime.datetime.fromtimestamp(sunset_unix).strftime('%H:%M:%S')
        if sunset_unix
        else "N/A"
    )

    processed_data = {
        "city_name": city_name,
        "description": main_weather,
        "temperature_c": temp,
        "feels_like_c": feels_like,
        "humidity_pct": humidity,
        "pressure_hpa": pressure,
        "wind_speed_ms": wind_speed,
        "wind_direction_deg": wind_deg,
        "cloudiness_pct": cloudiness,
        "sunrise": sunrise_time,
        "sunset": sunset_time,
    }

    return processed_data


# --------------------------- Prompt Building --------------------------------- #

def build_user_prompt(weather_data: dict) -> str:
    """
    Constructs a user-level prompt combining the processed weather information.
    The LLM can then use these details to create a dramatic, poetic interpretation.

    :param weather_data: Dictionary of processed weather info.
    :return: String containing a detailed user prompt.
    """
    if not weather_data:
        return "No valid weather data found. Please provide correct data."

    user_prompt = (
        f"City: {weather_data['city_name']}\n"
        f"Weather Description: {weather_data['description']}\n"
        f"Temperature: {weather_data['temperature_c']} °C\n"
        f"Feels Like: {weather_data['feels_like_c']} °C\n"
        f"Humidity: {weather_data['humidity_pct']}%\n"
        f"Pressure: {weather_data['pressure_hpa']} hPa\n"
        f"Wind Speed: {weather_data['wind_speed_ms']} m/s\n"
        f"Wind Direction: {weather_data['wind_direction_deg']}°\n"
        f"Cloudiness: {weather_data['cloudiness_pct']}%\n"
        f"Sunrise: {weather_data['sunrise']}\n"
        f"Sunset: {weather_data['sunset']}\n\n"
        "Create a theatrical, poetic interpretation of this atmospheric state. "
        "Reference the temperature, humidity, wind, cloudiness, sunrise, or sunset—"
        "imagine them as characters on a grand cosmic stage. Keep it elegant, "
        "imaginative, yet grounded in the facts above."
    )

    return user_prompt


# --------------------------- LLM Interaction --------------------------------- #

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Sends the system prompt and user prompt to the OpenAI ChatCompletion endpoint
    (GPT-3.5, GPT-4, etc.). Returns the AI’s response as a string.

    :param system_prompt: Overall directive setting the style or tone.
    :param user_prompt: Data-driven prompt with specific weather details.
    :return: AI-generated text describing the weather in a poetic or dramatic form.
    """
    openai.api_key = OPENAI_API_KEY

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,  # Increase for more creativity, decrease for more factual text
            max_tokens=600,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM request failed. Error: {str(e)}"


# --------------------------- Main Orchestration ------------------------------ #

def main():
    """
    Orchestrates the entire Weather-Quill script:
    1. Prompts the user for a city name.
    2. Fetches raw weather data from OpenWeatherMap.
    3. Processes that data into a structured format.
    4. Builds an LLM-ready user prompt.
    5. Calls the LLM to get a theatrical weather narrative.
    6. Prints out the result.
    """
    city = input("Enter a city name for the grand meteorological opera: ")

    # Step 1: Fetch raw data
    raw_data = fetch_weather_data(city, OPENWEATHER_API_KEY)
    if not raw_data:
        print("No data returned from the API. Exiting...")
        return

    # Step 2: Process the weather data
    processed = process_weather_data(raw_data)
    if not processed:
        print("Weather data was invalid. Exiting...")
        return

    # Step 3: Build the user prompt
    user_prompt = build_user_prompt(processed)

    # Step 4: Call the LLM for a theatrical interpretation
    theatrical_report = call_llm(SYSTEM_PROMPT, user_prompt)

    # Step 5: Print the output
    print("\n====== Theatrical Weather Report ======\n")
    print(theatrical_report)
    print("\n====== End of Report ======\n")


if __name__ == "__main__":
    main()
