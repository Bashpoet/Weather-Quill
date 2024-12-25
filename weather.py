import requests
import openai
import datetime

# If you have environment variables or a .env file, you can import os and dotenv here.
# import os
# from dotenv import load_dotenv
# load_dotenv()

# ---------- Configuration Section ----------

# Replace with your actual OpenWeatherMap key
OPENWEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"

# Replace with your actual OpenAI API key
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# We store them in code for simplicity; in production, definitely use environment variables:
# OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# This system prompt sets the overall style and "personality" of the LLM.
SYSTEM_PROMPT = """
You are a dramaturge of the atmosphere—a meteorological muse who weaves weather data into an immersive,
theatrical narrative. You adopt a playful, eloquent tone. Clouds, winds, humidity, and temperature are characters
on a grand stage. Pressure gradients conspire in the wings, humidity flirts at center stage, and the wind
acts as a gentle prompter or cunning director. Keep your commentary scientifically grounded but delivered in
lively, lyric prose. Reference the data you receive as if unveiling a hidden script that reveals the drama overhead.
"""


# ---------- Weather Data Retrieval and Processing ----------

def fetch_weather_data(city_name, api_key):
    """
    Retrieves current weather data from the OpenWeatherMap API for the specified city.

    :param city_name: The city for which to fetch weather data (e.g. "Paris", "Tokyo").
    :param api_key: Your OpenWeatherMap API key.
    :return: dict containing raw JSON weather data, or None if an error occurs.
    """
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_name,
        "appid": api_key,
        "units": "metric"  # We'll go ahead and get Celsius data directly here.
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # This will throw an error for non-2xx status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: Unable to retrieve data from OpenWeatherMap. Details: {e}")
        return None


def process_weather_data(raw_data):
    """
    Extracts and packages relevant weather information from raw JSON data into a tidy dict.

    :param raw_data: Raw JSON from the OpenWeatherMap API.
    :return: dict containing processed weather info, or None if parsing fails.
    """
    if not raw_data:
        return None

    city_name = raw_data.get("name", "Unknown City")
    weather_conditions = raw_data.get("weather", [{}])
    main_weather = weather_conditions[0].get("description", "No description")

    main_data = raw_data.get("main", {})
    temp = main_data.get("temp", 0.0)          # Celsius
    feels_like = main_data.get("feels_like", 0.0)
    humidity = main_data.get("humidity", 0)
    pressure = main_data.get("pressure", 0)

    wind_data = raw_data.get("wind", {})
    wind_speed = wind_data.get("speed", 0.0)   # m/s
    wind_deg = wind_data.get("deg", 0)

    clouds_data = raw_data.get("clouds", {})
    cloudiness = clouds_data.get("all", 0)

    # Optional extras: sunrise, sunset, coordinates
    sys_data = raw_data.get("sys", {})
    sunrise_unix = sys_data.get("sunrise")
    sunset_unix = sys_data.get("sunset")

    # Convert sunrise/sunset from UNIX timestamp to something human-readable
    # if they exist:
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

    # Return a dictionary capturing the relevant details
    processed = {
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

    return processed


# ---------- LLM Interaction ----------

def build_user_prompt(weather_data):
    """
    Constructs a user prompt from the processed weather data, referencing each piece of information
    in a straightforward way, so that the LLM can spin it into a theatrical narrative.

    :param weather_data: dict with processed weather info (see `process_weather_data`).
    :return: A string that the LLM can interpret to produce creative output.
    """
    if not weather_data:
        return "No valid weather data found. Please provide correct data."

    # We assemble the user prompt with just enough structure to let the LLM flourish:
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
    )
    user_prompt += (
        "Create a theatrical, poetic interpretation of this atmospheric state. "
        "Reference the temperature, humidity, wind, cloudiness, sunrise or sunset—"
        "imagine them as characters on a grand cosmic stage. "
        "Keep it elegant, imaginative, yet grounded in the facts above."
    )

    return user_prompt


def call_llm(system_prompt, user_prompt):
    """
    Sends the prompts to OpenAI's ChatCompletion endpoint (GPT-3.5, GPT-4, etc.).
    Adjust the model name and parameters as desired.

    :param system_prompt: The 'system' message that sets the high-level style/tone.
    :param user_prompt: The 'user' message containing weather data in a textual format.
    :return: str, the response from the LLM.
    """
    openai.api_key = OPENAI_API_KEY

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,    # Adjust for more or less creativity
            max_tokens=600,    # Adjust based on how long you want the response
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM request failed. Error: {str(e)}"


# ---------- Main Orchestration ----------

def main():
    # Step 1: Prompt user for a city
    city = input("Enter a city name for the grand meteorological opera: ")

    # Step 2: Fetch raw weather data
    raw_data = fetch_weather_data(city, OPENWEATHER_API_KEY)
    if not raw_data:
        print("No data returned from the API. Exiting...")
        return

    # Step 3: Process the raw data into a more coherent structure
    processed = process_weather_data(raw_data)
    if not processed:
        print("Weather data was invalid. Exiting...")
        return

    # Step 4: Build a user prompt with the processed data
    user_prompt = build_user_prompt(processed)

    # Step 5: Call the LLM with system and user prompts
    theatrical_report = call_llm(SYSTEM_PROMPT, user_prompt)

    # Step 6: Print out the flamboyant weather report
    print("\n====== Theatrical Weather Report ======\n")
    print(theatrical_report)
    print("\n====== End of Report ======\n")


if __name__ == "__main__":
    main()
