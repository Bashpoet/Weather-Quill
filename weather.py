import os
import requests
import openai
import datetime

# --------------------------- Configuration Section --------------------------- #

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHERMAP_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are a dramaturge of the atmosphere—a meteorological muse who weaves weather data into an immersive,
theatrical narrative. You adopt a playful, eloquent tone. Clouds, winds, humidity, and temperature are characters
on a grand stage. Pressure gradients conspire in the wings, humidity flirts at center stage, and the wind
acts as a gentle prompter or cunning director. Keep your commentary scientifically grounded but delivered in
lively, lyrical prose. Reference the data you receive as if unveiling a hidden script that reveals the drama overhead.
"""

# --------------------------- Utility Functions ------------------------------- #

def deg_to_compass(deg):
    """Convert wind direction in degrees to a compass direction."""
    if deg is None:
        return "N/A"
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(deg / 22.5) % 16
    return directions[index]

def convert_to_local_time(unix_time, offset):
    """Convert UTC Unix timestamp to local time string using timezone offset."""
    if unix_time is None:
        return "N/A"
    utc_time = datetime.datetime.utcfromtimestamp(unix_time)
    local_time = utc_time + datetime.timedelta(seconds=offset)
    return local_time.strftime('%Y-%m-%d %H:%M:%S')

# --------------------------- Weather Data Retrieval -------------------------- #

def fetch_weather_data(city_name: str, api_key: str) -> dict:
    """
    Fetch current weather data from OpenWeatherMap for a specified city.
    Returns raw JSON data or None on failure with specific error messages.
    """
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city_name, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            print("City not found. Please check the spelling or try another city.")
        elif response.status_code == 401:
            print("Invalid API key. Please check your OpenWeatherMap API key.")
        else:
            print(f"Error {response.status_code}: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None

# --------------------------- Data Processing --------------------------------- #

def process_weather_data(raw_data: dict) -> dict:
    """
    Process raw JSON weather data into a structured dictionary, adjusting times to local time zone.
    """
    if not raw_data:
        return None

    city_name = raw_data.get("name", "Unknown City")
    weather_array = raw_data.get("weather", [{}])
    main_weather = weather_array[0].get("description", "No description")
    main_data = raw_data.get("main", {})
    temp = main_data.get("temp", 0.0)
    feels_like = main_data.get("feels_like", 0.0)
    humidity = main_data.get("humidity", 0)
    pressure = main_data.get("pressure", 0)
    wind_data = raw_data.get("wind", {})
    wind_speed = wind_data.get("speed", 0.0)
    wind_deg = wind_data.get("deg", 0)
    cloud_data = raw_data.get("clouds", {})
    cloudiness = cloud_data.get("all", 0)
    sys_data = raw_data.get("sys", {})
    sunrise_unix = sys_data.get("sunrise")
    sunset_unix = sys_data.get("sunset")
    data_time_unix = raw_data.get("dt")
    timezone_offset = raw_data.get("timezone", 0)  # Seconds from UTC

    # Convert times to local time
    data_time = convert_to_local_time(data_time_unix, timezone_offset)
    sunrise_time = convert_to_local_time(sunrise_unix, timezone_offset)
    sunset_time = convert_to_local_time(sunset_unix, timezone_offset)
    wind_direction = deg_to_compass(wind_deg)

    return {
        "city_name": city_name,
        "description": main_weather,
        "temperature_c": temp,
        "feels_like_c": feels_like,
        "humidity_pct": humidity,
        "pressure_hpa": pressure,
        "wind_speed_ms": wind_speed,
        "wind_direction": wind_direction,
        "wind_direction_deg": wind_deg,
        "cloudiness_pct": cloudiness,
        "sunrise": sunrise_time,
        "sunset": sunset_time,
        "data_time": data_time
    }

# --------------------------- Prompt Building --------------------------------- #

def build_user_prompt(weather_data: dict) -> str:
    """Build a contextual prompt for the LLM with local time and compass wind direction."""
    if not weather_data:
        return "No valid weather data found. Please provide correct data."

    user_prompt = (
        f"As of {weather_data['data_time']}, in the city of {weather_data['city_name']},\n"
        f"Weather Description: {weather_data['description']}\n"
        f"Temperature: {weather_data['temperature_c']} °C\n"
        f"Feels Like: {weather_data['feels_like_c']} °C\n"
        f"Humidity: {weather_data['humidity_pct']}%\n"
        f"Pressure: {weather_data['pressure_hpa']} hPa\n"
        f"Wind Speed: {weather_data['wind_speed_ms']} m/s\n"
        f"Wind Direction: {weather_data['wind_direction']}\n"
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
    """Send prompts to OpenAI API and return the poetic response."""
    openai.api_key = OPENAI_API_KEY
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
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
    """Orchestrate the Weather-Quill script with improved functionality."""
    city = input("Enter a city name for the grand meteorological opera: ")
    raw_data = fetch_weather_data(city, OPENWEATHER_API_KEY)
    if not raw_data:
        print("No data returned from the API. Exiting...")
        return

    processed = process_weather_data(raw_data)
    if not processed:
        print("Weather data was invalid. Exiting...")
        return

    user_prompt = build_user_prompt(processed)
    theatrical_report = call_llm(SYSTEM_PROMPT, user_prompt)

    print("\n====== Theatrical Weather Report ======\n")
    print(theatrical_report)
    print("\n====== End of Report ======\n")

if __name__ == "__main__":
    main()
