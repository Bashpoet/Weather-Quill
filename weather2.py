Weather-Quill-v2.py

import os
import sys
import json
import time
import argparse
import logging
import requests
import datetime
import traceback
import tempfile
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from abc import ABC, abstractmethod

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import Anthropic/Claude
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Try to import Flask
try:
    from flask import Flask, render_template, request, send_file, jsonify, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Try to import geopy
try:
    import geopy.geocoders
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

# Try to import gTTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# --------------------------- Configuration & Setup --------------------------- #

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("weather-quill")

# Default configuration
DEFAULT_CONFIG = {
    "openweather_url": "http://api.openweathermap.org/data/2.5/weather",
    "openweather_forecast_url": "http://api.openweathermap.org/data/2.5/forecast",
    "openweather_history_url": "http://api.openweathermap.org/data/2.5/onecall/timemachine",
    "visualcrossing_url": "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline",
    "weatherapi_url": "http://api.weatherapi.com/v1",
    
    # LLM configuration
    "llm_provider": "openai",  # Default provider
    "openai_model": "gpt-3.5-turbo",
    "claude_model": "claude-3-opus-20240229",
    "temperature": 0.9,
    "max_tokens": 600,
    
    # Other settings
    "units": "metric",
    "cache_ttl": 600,  # 10 minutes cache TTL
    "output_style": "dramatic",  # default style
    "web_port": 5000,
    "web_host": "127.0.0.1",
    "reports_dir": "reports",
    "audio_dir": "audio",
    "geolocation_enabled": True
}

# Available styles for the LLM output
STYLE_PROMPTS = {
    "dramatic": """
        You are a dramaturge of the atmosphere—a meteorological muse who weaves weather data into an immersive,
        theatrical narrative. You adopt a playful, eloquent tone. Clouds, winds, humidity, and temperature are characters
        on a grand stage. Pressure gradients conspire in the wings, humidity flirts at center stage, and the wind
        acts as a gentle prompter or cunning director. Keep your commentary scientifically grounded but delivered in
        lively, lyrical prose. Reference the data you receive as if unveiling a hidden script that reveals the drama overhead.
    """,
    "shakespearean": """
        You are a Shakespearean bard of meteorology. Craft a sonnet or soliloquy about the current weather,
        using iambic pentameter where possible. Include weather imagery and metaphors that Shakespeare
        might have used. Reference the specific weather data in your verse, but transform it into
        a tale of cosmic significance. 'What light through yonder atmosphere breaks?'
    """,
    "noir": """
        You are a hardboiled detective of meteorology. The city is your beat, and the weather is your case.
        Narrate the current atmospheric conditions in the style of Raymond Chandler or Dashiell Hammett.
        The temperature is suspicious, the clouds have motives, and the wind might be hiding something.
        Keep it gritty, terse, and cynical, but accurate to the weather data provided.
    """,
    "scientific": """
        You are a poetic scientist explaining weather phenomena. While maintaining scientific accuracy,
        craft an elegant explanation of the current atmospheric conditions that would inspire wonder and
        curiosity. Weave metaphors and imagery that illuminate the physics and chemistry at work,
        making the invisible forces of meteorology tangible and beautiful to contemplate.
    """,
    "melodramatic": """
        You are a WILDLY DRAMATIC meteorologist who sees COSMIC SIGNIFICANCE in every cloud! Each shift 
        in the wind is a SHOCKING TWIST in the GRAND NARRATIVE of the SKY! Use excessive exclamation points,
        DRAMATIC CAPITALIZATION, and breathless, over-the-top language to describe even the most mundane 
        weather conditions. Make the audience feel they are witnessing the MOST IMPORTANT ATMOSPHERIC EVENT 
        OF ALL TIME!
    """,
    "grim": """
        You are a somber herald of atmospheric conditions. The weather is but a reflection of life's inevitable 
        march toward entropy. Speak of clouds as harbingers of doom, of sunlight as fleeting moments in an 
        uncaring universe. Use gothic, melancholic language reminiscent of Edgar Allan Poe or H.P. Lovecraft. 
        The weather report should feel like a meditation on mortality and the cosmic indifference of nature.
    """,
    "children": """
        You are a friendly, whimsical storyteller explaining weather to curious children. The clouds are 
        fluffy characters with personalities, the wind is playful and mischievous, and the rain is just 
        the sky sharing its tears of joy. Use simple language, gentle humor, and a sense of wonder. Make 
        meteorological concepts accessible and fun for young minds, while maintaining scientific accuracy 
        in child-friendly terms.
    """,
    "historical": """
        You are a time-traveling weather chronicler. For historical weather, describe it as if you were 
        a person from that era writing in their journal or newspaper. Use period-appropriate language, 
        references, and context. For instance, weather from 1923 should be described with 1920s terminology 
        and cultural references. Make connections to historical events happening around that time if relevant.
    """
}

# --------------------------- Data Models ------------------------------------ #

@dataclass
class GeoContext:
    """Geographic context information about a location"""
    country: Optional[str] = None
    region: Optional[str] = None
    is_coastal: bool = False
    elevation: Optional[float] = None
    nearby_water: Optional[str] = None
    terrain: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def to_description(self) -> str:
        """Generate a textual description of the geographic context"""
        parts = []
        
        if self.region and self.country:
            parts.append(f"in the {self.region} region of {self.country}")
        elif self.country:
            parts.append(f"in {self.country}")
            
        if self.is_coastal:
            parts.append("along the coast")
        
        if self.nearby_water:
            parts.append(f"near the {self.nearby_water}")
            
        if self.elevation and self.elevation > 1000:
            parts.append(f"at an elevation of {self.elevation:.0f} meters")
            
        if self.terrain:
            parts.append(f"in {self.terrain} terrain")
            
        if not parts:
            return ""
            
        return ", ".join(parts)

@dataclass
class WeatherData:
    """Structured weather data class"""
    city_name: str
    description: str
    temperature_c: float
    feels_like_c: float
    humidity_pct: int
    pressure_hpa: int
    wind_speed_ms: float
    wind_direction: str
    wind_direction_deg: int
    cloudiness_pct: int
    sunrise: str
    sunset: str
    data_time: str
    provider: str = "unknown"
    uv_index: Optional[float] = None
    precipitation_mm: Optional[float] = None
    air_quality_index: Optional[int] = None
    historical_date: Optional[str] = None
    geo_context: Optional[GeoContext] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass to a dictionary for serialization"""
        result = asdict(self)
        # Convert nested dataclass if present
        if self.geo_context:
            result['geo_context'] = self.geo_context.to_dict()
        return result
    
    def to_fahrenheit(self) -> Tuple[float, float]:
        """Convert temperatures to Fahrenheit"""
        temp_f = (self.temperature_c * 9/5) + 32
        feels_like_f = (self.feels_like_c * 9/5) + 32
        return temp_f, feels_like_f
    
    def temperature_formatted(self, units: str = "metric") -> str:
        """Get formatted temperature string in requested units"""
        if units == "imperial":
            temp_f, feels_f = self.to_fahrenheit()
            return f"{temp_f:.1f}°F (feels like {feels_f:.1f}°F)"
        return f"{self.temperature_c:.1f}°C (feels like {self.feels_like_c:.1f}°C)"
    
    def wind_formatted(self, units: str = "metric") -> str:
        """Get formatted wind string in requested units"""
        if units == "imperial":
            wind_mph = self.wind_speed_ms * 2.237
            return f"{wind_mph:.1f} mph from {self.wind_direction}"
        return f"{self.wind_speed_ms:.1f} m/s from {self.wind_direction}"

@dataclass
class ForecastData:
    """Container for forecast data"""
    city_name: str
    forecast_days: List[WeatherData]
    provider: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "city_name": self.city_name,
            "provider": self.provider,
            "forecast_days": [day.to_dict() for day in self.forecast_days]
        }

class WeatherCache:
    """Cache for weather data to minimize API calls"""
    def __init__(self, ttl: int = 600):
        self.cache = {}
        self.ttl = ttl
        self.cache_file = Path.home() / ".weather_quill_cache.json"
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from file if it exists"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Filter out expired entries
                    current_time = time.time()
                    self.cache = {
                        k: v for k, v in cache_data.items() 
                        if current_time - v.get('timestamp', 0) < self.ttl
                    }
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except IOError as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get an item from cache if it exists and is not expired"""
        if key in self.cache:
            entry = self.cache[key]
            current_time = time.time()
            if current_time - entry.get('timestamp', 0) < self.ttl:
                return entry.get('data')
            else:
                # Remove expired entry
                del self.cache[key]
                self._save_cache()
        return None
    
    def set(self, key: str, data: Dict[str, Any]) -> None:
        """Set an item in the cache with current timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        self._save_cache()

# --------------------------- LLM Provider Interface ------------------------ #

class LLMProvider(ABC):
    """Abstract base class for language model providers"""
    
    @abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str, 
                     temperature: float = 0.9, max_tokens: int = 600) -> str:
        """Generate text based on prompts"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models from this provider"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI (GPT) implementation"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """Initialize with API key and model"""
        self.api_key = api_key
        self.model = model
        
    def get_provider_name(self) -> str:
        return "OpenAI"
    
    def get_available_models(self) -> List[str]:
        return [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o" 
        ]
    
    def generate_text(self, system_prompt: str, user_prompt: str, 
                     temperature: float = 0.9, max_tokens: int = 600) -> str:
        """Generate text using OpenAI's API"""
        if not OPENAI_AVAILABLE:
            return "OpenAI Python package is not installed. Please install it with 'pip install openai'."
            
        try:
            import openai
            openai.api_key = self.api_key
            
            logger.info(f"Calling OpenAI with model: {self.model}")
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI request failed: {str(e)}")
            return f"OpenAI request failed. Error: {str(e)}"

class ClaudeProvider(LLMProvider):
    """Anthropic (Claude) implementation"""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        """Initialize with API key and model"""
        self.api_key = api_key
        self.model = model
        
    def get_provider_name(self) -> str:
        return "Claude"
    
    def get_available_models(self) -> List[str]:
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3.5-sonnet-20240620",
            "claude-3.7-sonnet-20250219"
        ]
    
    def generate_text(self, system_prompt: str, user_prompt: str, 
                     temperature: float = 0.9, max_tokens: int = 600) -> str:
        """Generate text using Anthropic's Claude API"""
        if not ANTHROPIC_AVAILABLE:
            return "Anthropic Python package is not installed. Please install it with 'pip install anthropic'."
            
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            
            logger.info(f"Calling Claude with model: {self.model}")
            
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude request failed: {str(e)}")
            return f"Claude request failed. Error: {str(e)}"

class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(provider_name: str, model: Optional[str] = None) -> Optional[LLMProvider]:
        """Create a provider by name"""
        provider_name = provider_name.lower()
        
        if provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OpenAI API key not set")
                return None
                
            model_to_use = model if model else DEFAULT_CONFIG["openai_model"]
            return OpenAIProvider(api_key, model_to_use)
            
        elif provider_name == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("Anthropic API key not set")
                return None
                
            model_to_use = model if model else DEFAULT_CONFIG["claude_model"]
            return ClaudeProvider(api_key, model_to_use)
            
        else:
            logger.error(f"Unknown LLM provider: {provider_name}")
            return None

# --------------------------- Weather Provider Interface --------------------- #

class WeatherProvider(ABC):
    """Abstract base class for weather data providers"""
    
    @abstractmethod
    def get_current_weather(self, location: str, units: str = "metric") -> Optional[WeatherData]:
        """Get current weather for location"""
        pass
    
    @abstractmethod
    def get_forecast(self, location: str, days: int = 5, units: str = "metric") -> Optional[ForecastData]:
        """Get forecast for location"""
        pass
    
    @abstractmethod
    def get_historical_weather(self, location: str, date: datetime.date, units: str = "metric") -> Optional[WeatherData]:
        """Get historical weather for location on date"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider"""
        pass

class OpenWeatherMapProvider(WeatherProvider):
    """OpenWeatherMap implementation"""
    
    def __init__(self, api_key: str, cache: Optional[WeatherCache] = None):
        """Initialize with API key and optional cache"""
        self.api_key = api_key
        self.cache = cache
        self.current_url = DEFAULT_CONFIG["openweather_url"]
        self.forecast_url = DEFAULT_CONFIG["openweather_forecast_url"]
        self.history_url = DEFAULT_CONFIG["openweather_history_url"]
    
    def get_provider_name(self) -> str:
        return "OpenWeatherMap"
    
    def get_current_weather(self, location: str, units: str = "metric") -> Optional[WeatherData]:
        """Get current weather from OpenWeatherMap"""
        # Check cache first
        if self.cache:
            cache_key = f"owm_current_{location}_{units}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved weather data for {location} from cache")
                return self._process_current_weather(cached_data)
        
        # Fetch from API
        params = {"q": location, "appid": self.api_key, "units": units}
        
        try:
            logger.info(f"Fetching current weather for {location} from OpenWeatherMap")
            response = requests.get(self.current_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Store in cache if available
                if self.cache:
                    cache_key = f"owm_current_{location}_{units}"
                    self.cache.set(cache_key, data)
                
                return self._process_current_weather(data)
            else:
                self._handle_error_response(response)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            return None
    
    def get_forecast(self, location: str, days: int = 5, units: str = "metric") -> Optional[ForecastData]:
        """Get forecast from OpenWeatherMap"""
        # Check cache first
        if self.cache:
            cache_key = f"owm_forecast_{location}_{units}_{days}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved forecast data for {location} from cache")
                return self._process_forecast(cached_data, days)
        
        # Fetch from API - OpenWeatherMap returns 5-day forecast in 3-hour steps
        params = {"q": location, "appid": self.api_key, "units": units, "cnt": min(days * 8, 40)}
        
        try:
            logger.info(f"Fetching forecast for {location} from OpenWeatherMap")
            response = requests.get(self.forecast_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Store in cache if available
                if self.cache:
                    cache_key = f"owm_forecast_{location}_{units}_{days}"
                    self.cache.set(cache_key, data)
                
                return self._process_forecast(data, days)
            else:
                self._handle_error_response(response)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            return None
    
    def get_historical_weather(self, location: str, date: datetime.date, units: str = "metric") -> Optional[WeatherData]:
        """Get historical weather from OpenWeatherMap"""
        # OpenWeatherMap requires coordinates for historical data
        coords = self._get_coordinates(location)
        if not coords:
            logger.error(f"Could not get coordinates for {location}")
            return None
            
        lat, lon = coords
        
        # Check cache first
        if self.cache:
            cache_key = f"owm_historical_{location}_{date.isoformat()}_{units}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved historical data for {location} on {date} from cache")
                return self._process_historical(cached_data, location, date)
        
        # Convert date to timestamp
        dt = int(datetime.datetime.combine(date, datetime.time(12, 0)).timestamp())
        
        # Fetch from API
        params = {
            "lat": lat, 
            "lon": lon, 
            "dt": dt,
            "appid": self.api_key, 
            "units": units
        }
        
        try:
            logger.info(f"Fetching historical weather for {location} on {date} from OpenWeatherMap")
            response = requests.get(self.history_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Store in cache if available
                if self.cache:
                    cache_key = f"owm_historical_{location}_{date.isoformat()}_{units}"
                    self.cache.set(cache_key, data)
                
                return self._process_historical(data, location, date)
            else:
                self._handle_error_response(response)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            return None
    
    def _get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location using OpenWeatherMap Geocoding API"""
        geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {"q": location, "limit": 1, "appid": self.api_key}
        
        try:
            response = requests.get(geocoding_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]["lat"], data[0]["lon"]
            
            return None
                
        except requests.exceptions.RequestException:
            return None
    
    def _process_current_weather(self, data: Dict[str, Any]) -> WeatherData:
        """Process raw current weather data from OpenWeatherMap API"""
        city_name = data.get("name", "Unknown City")
        weather_array = data.get("weather", [{}])
        main_weather = weather_array[0].get("description", "No description") if weather_array else "No description"
        main_data = data.get("main", {})
        temp = main_data.get("temp", 0.0)
        feels_like = main_data.get("feels_like", 0.0)
        humidity = main_data.get("humidity", 0)
        pressure = main_data.get("pressure", 0)
        wind_data = data.get("wind", {})
        wind_speed = wind_data.get("speed", 0.0)
        wind_deg = wind_data.get("deg")
        cloud_data = data.get("clouds", {})
        cloudiness = cloud_data.get("all", 0)
        sys_data = data.get("sys", {})
        sunrise_unix = sys_data.get("sunrise")
        sunset_unix = sys_data.get("sunset")
        data_time_unix = data.get("dt")
        timezone_offset = data.get("timezone", 0)  # Seconds from UTC

        # Convert times to local time
        data_time = convert_to_local_time(data_time_unix, timezone_offset)
        sunrise_time = convert_to_local_time(sunrise_unix, timezone_offset)
        sunset_time = convert_to_local_time(sunset_unix, timezone_offset)
        wind_direction = deg_to_compass(wind_deg)
        
        # Get precipitation if available (OpenWeatherMap doesn't always provide this)
        rain_data = data.get("rain", {})
        snow_data = data.get("snow", {})
        precipitation_mm = rain_data.get("1h", 0.0) + snow_data.get("1h", 0.0)

        return WeatherData(
            city_name=city_name,
            description=main_weather,
            temperature_c=temp,
            feels_like_c=feels_like,
            humidity_pct=humidity,
            pressure_hpa=pressure,
            wind_speed_ms=wind_speed,
            wind_direction=wind_direction,
            wind_direction_deg=wind_deg if wind_deg is not None else 0,
            cloudiness_pct=cloudiness,
            sunrise=sunrise_time,
            sunset=sunset_time,
            data_time=data_time,
            provider=self.get_provider_name(),
            precipitation_mm=precipitation_mm if precipitation_mm > 0 else None
        )
    
    def _process_forecast(self, data: Dict[str, Any], days: int) -> ForecastData:
        """Process raw forecast data from OpenWeatherMap API"""
        city_data = data.get("city", {})
        city_name = city_data.get("name", "Unknown City")
        timezone_offset = city_data.get("timezone", 0)
        
        # Group forecast by day
        daily_forecasts = {}
        
        for item in data.get("list", []):
            # Convert timestamp to date
            dt = item.get("dt")
            if not dt:
                continue
                
            date = datetime.datetime.utcfromtimestamp(dt).date()
            
            # Skip if we already have enough days
            if len(daily_forecasts) >= days and date not in daily_forecasts:
                continue
                
            # Initialize day data or update with this time slot if it's better (noon is preferred)
            hour = datetime.datetime.utcfromtimestamp(dt).hour
            if date not in daily_forecasts or abs(hour - 12) < abs(daily_forecasts[date]["hour"] - 12):
                weather_array = item.get("weather", [{}])
                main_weather = weather_array[0].get("description", "No description") if weather_array else "No description"
                main_data = item.get("main", {})
                
                daily_forecasts[date] = {
                    "hour": hour,
                    "description": main_weather,
                    "temperature_c": main_data.get("temp", 0.0),
                    "feels_like_c": main_data.get("feels_like", 0.0),
                    "humidity_pct": main_data.get("humidity", 0),
                    "pressure_hpa": main_data.get("pressure", 0),
                    "wind_speed_ms": item.get("wind", {}).get("speed", 0.0),
                    "wind_deg": item.get("wind", {}).get("deg", 0),
                    "cloudiness_pct": item.get("clouds", {}).get("all", 0),
                    "data_time": convert_to_local_time(dt, timezone_offset),
                    "precipitation_mm": item.get("rain", {}).get("3h", 0.0) + item.get("snow", {}).get("3h", 0.0)
                }
        
        # Convert to WeatherData objects
        forecast_days = []
        
        for date, day_data in sorted(daily_forecasts.items()):
            # Use sunrise/sunset from current day as approximation
            forecast_days.append(WeatherData(
                city_name=city_name,
                description=day_data["description"],
                temperature_c=day_data["temperature_c"],
                feels_like_c=day_data["feels_like_c"],
                humidity_pct=day_data["humidity_pct"],
                pressure_hpa=day_data["pressure_hpa"],
                wind_speed_ms=day_data["wind_speed_ms"],
                wind_direction=deg_to_compass(day_data["wind_deg"]),
                wind_direction_deg=day_data["wind_deg"],
                cloudiness_pct=day_data["cloudiness_pct"],
                sunrise="N/A",  # Not available in forecast
                sunset="N/A",   # Not available in forecast
                data_time=day_data["data_time"],
                provider=self.get_provider_name(),
                precipitation_mm=day_data["precipitation_mm"] if day_data["precipitation_mm"] > 0 else None
            ))
        
        return ForecastData(
            city_name=city_name,
            forecast_days=forecast_days,
            provider=self.get_provider_name()
        )
    
    def _process_historical(self, data: Dict[str, Any], location: str, date: datetime.date) -> WeatherData:
        """Process raw historical data from OpenWeatherMap API"""
        # Historical API returns different format
        current_data = data.get("current", {})
        weather_array = current_data.get("weather", [{}])
        main_weather = weather_array[0].get("description", "No description") if weather_array else "No description"
        
        return WeatherData(
            city_name=location,
            description=main_weather,
            temperature_c=current_data.get("temp", 0.0),
            feels_like_c=current_data.get("feels_like", 0.0),
            humidity_pct=current_data.get("humidity", 0),
            pressure_hpa=current_data.get("pressure", 0),
            wind_speed_ms=current_data.get("wind_speed", 0.0),
            wind_direction=deg_to_compass(current_data.get("wind_deg")),
            wind_direction_deg=current_data.get("wind_deg", 0),
            cloudiness_pct=current_data.get("clouds", 0),
            sunrise=convert_to_local_time(current_data.get("sunrise"), data.get("timezone_offset", 0)),
            sunset=convert_to_local_time(current_data.get("sunset"), data.get("timezone_offset", 0)),
            data_time=convert_to_local_time(current_data.get("dt"), data.get("timezone_offset", 0)),
            provider=self.get_provider_name(),
            uv_index=current_data.get("uvi"),
            historical_date=date.isoformat()
        )
    
    def _handle_error_response(self, response: requests.Response) -> None:
        """Handle error responses from the API"""
        if response.status_code == 404:
            logger.error("Location not found. Please check the spelling or try another location.")
        elif response.status_code == 401:
            logger.error("Invalid API key. Please check your OpenWeatherMap API key.")
        else:
            logger.error(f"Error {response.status_code}: {response.text}")

class WeatherAPIProvider(WeatherProvider):
    """WeatherAPI.com implementation"""
    
    def __init__(self, api_key: str, cache: Optional[WeatherCache] = None):
        """Initialize with API key and optional cache"""
        self.api_key = api_key
        self.cache = cache
        self.base_url = DEFAULT_CONFIG["weatherapi_url"]
    
    def get_provider_name(self) -> str:
        return "WeatherAPI"
    
    def get_current_weather(self, location: str, units: str = "metric") -> Optional[WeatherData]:
        """Get current weather from WeatherAPI.com"""
        # Check cache first
        if self.cache:
            cache_key = f"wapi_current_{location}_{units}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved weather data for {location} from cache")
                return self._process_current_weather(cached_data)
        
        # Fetch from API
        url = f"{self.base_url}/current.json"
        params = {"key": self.api_key, "q": location, "aqi": "yes"}
        
        try:
            logger.info(f"Fetching current weather for {location} from WeatherAPI")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Store in cache if available
                if self.cache:
                    cache_key = f"wapi_current_{location}_{units}"
                    self.cache.set(cache_key, data)
                
                return self._process_current_weather(data)
            else:
                self._handle_error_response(response)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            return None
    
    def get_forecast(self, location: str, days: int = 5, units: str = "metric") -> Optional[ForecastData]:
        """Get forecast from WeatherAPI.com"""
        # Check cache first
        if self.cache:
            cache_key = f"wapi_forecast_{location}_{units}_{days}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved forecast data for {location} from cache")
                return self._process_forecast(cached_data)
        
        # Fetch from API
        url = f"{self.base_url}/forecast.json"
        params = {"key": self.api_key, "q": location, "days": days, "aqi": "yes"}
        
        try:
            logger.info(f"Fetching forecast for {location} from WeatherAPI")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Store in cache if available
                if self.cache:
                    cache_key = f"wapi_forecast_{location}_{units}_{days}"
                    self.cache.set(cache_key, data)
                
                return self._process_forecast(data)
            else:
                self._handle_error_response(response)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            return None
    
    def get_historical_weather(self, location: str, date: datetime.date, units: str = "metric") -> Optional[WeatherData]:
        """Get historical weather from WeatherAPI.com"""
        # Check cache first
        if self.cache:
            cache_key = f"wapi_historical_{location}_{date.isoformat()}_{units}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved historical data for {location} on {date} from cache")
                return self._process_historical(cached_data)
        
        # Format date as required by API
        date_str = date.strftime("%Y-%m-%d")
        
        # Fetch from API
        url = f"{self.base_url}/history.json"
        params = {"key": self.api_key, "q": location, "dt": date_str}
        
        try:
            logger.info(f"Fetching historical weather for {location} on {date} from WeatherAPI")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Store in cache if available
                if self.cache:
                    cache_key = f"wapi_historical_{location}_{date.isoformat()}_{units}"
                    self.cache.set(cache_key, data)
                
                return self._process_historical(data)
            else:
                self._handle_error_response(response)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            return None
    
    def _process_current_weather(self, data: Dict[str, Any]) -> WeatherData:
        """Process raw current weather data from WeatherAPI.com"""
        location_data = data.get("location", {})
        current_data = data.get("current", {})
        
        city_name = location_data.get("name", "Unknown City")
        
        # Convert time strings to local time format
        localtime_str = location_data.get("localtime", "")
        try:
            local_dt = datetime.datetime.strptime(localtime_str, "%Y-%m-%d %H:%M")
            data_time = local_dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            data_time = localtime_str
        
        # WeatherAPI doesn't provide sunrise/sunset in current API
        # We could get it from astronomy API if needed
        
        return WeatherData(
            city_name=city_name,
            description=current_data.get("condition", {}).get("text", "No description"),
            temperature_c=current_data.get("temp_c", 0.0),
            feels_like_c=current_data.get("feelslike_c", 0.0),
            humidity_pct=current_data.get("humidity", 0),
            pressure_hpa=current_data.get("pressure_mb", 0),
            wind_speed_ms=current_data.get("wind_kph", 0.0) / 3.6,  # Convert kph to m/s
            wind_direction=current_data.get("wind_dir", "N/A"),
            wind_direction_deg=current_data.get("wind_degree", 0),
            cloudiness_pct=current_data.get("cloud", 0),
            sunrise="N/A",  # Not available in this API call
            sunset="N/A",   # Not available in this API call
            data_time=data_time,
            provider=self.get_provider_name(),
            uv_index=current_data.get("uv", None),
            precipitation_mm=current_data.get("precip_mm", 0.0) if current_data.get("precip_mm", 0.0) > 0 else None,
            air_quality_index=current_data.get("air_quality", {}).get("us-epa-index")
        )
    
    def _process_forecast(self, data: Dict[str, Any]) -> ForecastData:
        """Process raw forecast data from WeatherAPI.com"""
        location_data = data.get("location", {})
        forecast_data = data.get("forecast", {})
        
        city_name = location_data.get("name", "Unknown City")
        forecast_days_data = forecast_data.get("forecastday", [])
        
        forecast_days = []
        
        for day_data in forecast_days_data:
            day = day_data.get("day", {})
            astro = day_data.get("astro", {})
            
            # Convert time strings
            date_str = day_data.get("date", "")
            sunrise_str = astro.get("sunrise", "")
            sunset_str = astro.get("sunset", "")
            
            forecast_days.append(WeatherData(
                city_name=city_name,
                description=day.get("condition", {}).get("text", "No description"),
                temperature_c=day.get("avgtemp_c", 0.0),
                feels_like_c=day.get("avgtemp_c", 0.0),  # WeatherAPI doesn't provide feels_like for forecast
                humidity_pct=day.get("avghumidity", 0),
                pressure_hpa=1013,  # Not provided in forecast
                wind_speed_ms=day.get("maxwind_kph", 0.0) / 3.6,  # Convert kph to m/s
                wind_direction="N/A",  # Not provided for daily forecast
                wind_direction_deg=0,  # Not provided for daily forecast
                cloudiness_pct=0,  # Not provided in forecast
                sunrise=sunrise_str,
                sunset=sunset_str,
                data_time=date_str,
                provider=self.get_provider_name(),
                uv_index=day.get("uv", None),
                precipitation_mm=day.get("totalprecip_mm", 0.0) if day.get("totalprecip_mm", 0.0) > 0 else None
            ))
        
        return ForecastData(
            city_name=city_name,
            forecast_days=forecast_days,
            provider=self.get_provider_name()
        )
    
    def _process_historical(self, data: Dict[str, Any]) -> WeatherData:
        """Process raw historical data from WeatherAPI.com"""
        location_data = data.get("location", {})
        forecast_data = data.get("forecast", {})
        
        city_name = location_data.get("name", "Unknown City")
        historical_days = forecast_data.get("forecastday", [])
        
        if not historical_days:
            logger.error("No historical data found")
            return None
            
        # Take the first (and should be only) day
        day_data = historical_days[0]
        day = day_data.get("day", {})
        astro = day_data.get("astro", {})
        
        # Convert time strings
        date_str = day_data.get("date", "")
        sunrise_str = astro.get("sunrise", "")
        sunset_str = astro.get("sunset", "")
        
        return WeatherData(
            city_name=city_name,
            description=day.get("condition", {}).get("text", "No description"),
            temperature_c=day.get("avgtemp_c", 0.0),
            feels_like_c=day.get("avgtemp_c", 0.0),  # Not provided for historical
            humidity_pct=day.get("avghumidity", 0),
            pressure_hpa=1013,  # Not provided for historical
            wind_speed_ms=day.get("maxwind_kph", 0.0) / 3.6,  # Convert kph to m/s
            wind_direction="N/A",  # Not provided for daily historical
            wind_direction_deg=0,  # Not provided for daily historical
            cloudiness_pct=0,  # Not provided in historical
            sunrise=sunrise_str,
            sunset=sunset_str,
            data_time=date_str,
            provider=self.get_provider_name(),
            uv_index=day.get("uv", None),
            precipitation_mm=day.get("totalprecip_mm", 0.0) if day.get("totalprecip_mm", 0.0) > 0 else None,
            historical_date=date_str
        )
    
    def _handle_error_response(self, response: requests.Response) -> None:
        """Handle error responses from the API"""
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", f"Error {response.status_code}")
            logger.error(error_msg)
        except:
            logger.error(f"Error {response.status_code}: {response.text}")

class VisualCrossingProvider(WeatherProvider):
    """Visual Crossing Weather API implementation"""
    
    def __init__(self, api_key: str, cache: Optional[WeatherCache] = None):
        """Initialize with API key and optional cache"""
        self.api_key = api_key
        self.cache = cache
        self.base_url = DEFAULT_CONFIG["visualcrossing_url"]
    
    def get_provider_name(self) -> str:
        return "VisualCrossing"
    
    def get_current_weather(self, location: str, units: str = "metric") -> Optional[WeatherData]:
        """Get current weather from Visual Crossing"""
        # Check cache first
        if self.cache:
            cache_key = f"vc_current_{location}_{units}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved weather data for {location} from cache")
                return self._process_timeline_data(cached_data, is_current=True)
        
        # Unit system for Visual Crossing
        unit_system = "metric" if units == "metric" else "us"
        
        # Fetch from API using timeline endpoint with today's date
        url = f"{self.base_url}/{location}/today"
        params = {
            "key": self.api_key,
            "unitGroup": unit_system,
            "include": "current",
            "contentType": "json"
        }
        
        try:
            logger.info(f"Fetching current weather for {location} from Visual Crossing")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Store in cache if available
                if self.cache:
                    cache_key = f"vc_current_{location}_{units}"
                    self.cache.set(cache_key, data)
                
                return self._process_timeline_data(data, is_current=True)
            else:
                self._handle_error_response(response)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            return None
    
    def get_forecast(self, location: str, days: int = 5, units: str = "metric") -> Optional[ForecastData]:
        """Get forecast from Visual Crossing"""
        # Check cache first
        if self.cache:
            cache_key = f"vc_forecast_{location}_{units}_{days}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved forecast data for {location} from cache")
                return self._process_forecast_data(cached_data, days)
        
        # Unit system for Visual Crossing
        unit_system = "metric" if units == "metric" else "us"
        
        # Construct date range for next N days
        today = datetime.date.today()
        end_date = today + datetime.timedelta(days=days-1)
        date_range = f"{today.isoformat()}/{end_date.isoformat()}"
        
        # Fetch from API
        url = f"{self.base_url}/{location}/{date_range}"
        params = {
            "key": self.api_key,
            "unitGroup": unit_system,
            "contentType": "json"
        }
        
        try:
            logger.info(f"Fetching forecast for {location} from Visual Crossing")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Store in cache if available
                if self.cache:
                    cache_key = f"vc_forecast_{location}_{units}_{days}"
                    self.cache.set(cache_key, data)
                
                return self._process_forecast_data(data, days)
            else:
                self._handle_error_response(response)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            return None
    
    def get_historical_weather(self, location: str, date: datetime.date, units: str = "metric") -> Optional[WeatherData]:
        """Get historical weather from Visual Crossing"""
        # Check cache first
        if self.cache:
            cache_key = f"vc_historical_{location}_{date.isoformat()}_{units}"
            cached_data = self.cache.get(cache_key)
            if cached_data:
                logger.info(f"Retrieved historical data for {location} on {date} from cache")
                return self._process_timeline_data(cached_data, is_historical=True, historical_date=date)
        
        # Unit system for Visual Crossing
        unit_system = "metric" if units == "metric" else "us"
        
        # Format date as required by API
        date_str = date.isoformat()
        
        # Fetch from API
        url = f"{self.base_url}/{location}/{date_str}"
        params = {
            "key": self.api_key,
            "unitGroup": unit_system,
            "contentType": "json"
        }
        
        try:
            logger.info(f"Fetching historical weather for {location} on {date} from Visual Crossing")
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Store in cache if available
                if self.cache:
                    cache_key = f"vc_historical_{location}_{date.isoformat()}_{units}"
                    self.cache.set(cache_key, data)
                
                return self._process_timeline_data(data, is_historical=True, historical_date=date)
            else:
                self._handle_error_response(response)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            return None
    
    def _process_timeline_data(self, data: Dict[str, Any], is_current: bool = False, 
                              is_historical: bool = False, historical_date: Optional[datetime.date] = None) -> WeatherData:
        """Process data from Visual Crossing Timeline API"""
        address = data.get("address", "Unknown Location")
        timezone = data.get("timezone", "UTC")
        
        # For current weather, use the current conditions
        # For historical, use the day's data
        if is_current and "currentConditions" in data:
            current = data["currentConditions"]
            description = current.get("conditions", "No description")
            temp = current.get("temp", 0.0)
            feels_like = current.get("feelslike", 0.0)
            humidity = current.get("humidity", 0)
            pressure = current.get("pressure", 0)
            wind_speed = current.get("windspeed", 0.0)
            wind_dir = current.get("winddir", 0)
            cloudiness = current.get("cloudcover", 0)
            uv_index = current.get("uvindex")
            precipitation = current.get("precip", 0.0)
            
            # Convert datetimeEpoch to human-readable format
            data_time_epoch = current.get("datetimeEpoch", 0)
            data_time = datetime.datetime.fromtimestamp(data_time_epoch).strftime('%Y-%m-%d %H:%M:%S')
            
            # Get sunrise and sunset
            sunrise_epoch = current.get("sunriseEpoch", 0)
            sunset_epoch = current.get("sunsetEpoch", 0)
            sunrise = datetime.datetime.fromtimestamp(sunrise_epoch).strftime('%H:%M:%S')
            sunset = datetime.datetime.fromtimestamp(sunset_epoch).strftime('%H:%M:%S')
            
        elif is_historical and "days" in data and len(data["days"]) > 0:
            # Find the day that matches our requested date
            requested_day = None
            for day in data["days"]:
                day_date = day.get("datetime", "")
                if historical_date and day_date == historical_date.isoformat():
                    requested_day = day
                    break
            
            # If we didn't find a matching day, use the first one
            if not requested_day and data["days"]:
                requested_day = data["days"][0]
                
            if not requested_day:
                logger.error("No historical data found for the requested date")
                return None
                
            description = requested_day.get("conditions", "No description")
            temp = requested_day.get("temp", 0.0)
            feels_like = requested_day.get("feelslike", 0.0)
            humidity = requested_day.get("humidity", 0)
            pressure = requested_day.get("pressure", 0)
            wind_speed = requested_day.get("windspeed", 0.0)
            wind_dir = requested_day.get("winddir", 0)
            cloudiness = requested_day.get("cloudcover", 0)
            uv_index = requested_day.get("uvindex")
            precipitation = requested_day.get("precip", 0.0)
            
            # Get date and time
            date_str = requested_day.get("datetime", "")
            data_time = f"{date_str} 12:00:00"  # Use noon as representative time
            
            # Get sunrise and sunset
            sunrise_epoch = requested_day.get("sunriseEpoch", 0)
            sunset_epoch = requested_day.get("sunsetEpoch", 0)
            sunrise = datetime.datetime.fromtimestamp(sunrise_epoch).strftime('%H:%M:%S')
            sunset = datetime.datetime.fromtimestamp(sunset_epoch).strftime('%H:%M:%S')
            
        else:
            # Fallback to days[0] if available
            if "days" in data and len(data["days"]) > 0:
                day = data["days"][0]
                description = day.get("conditions", "No description")
                temp = day.get("temp", 0.0)
                feels_like = day.get("feelslike", 0.0)
                humidity = day.get("humidity", 0)
                pressure = day.get("pressure", 0)
                wind_speed = day.get("windspeed", 0.0)
                wind_dir = day.get("winddir", 0)
                cloudiness = day.get("cloudcover", 0)
                uv_index = day.get("uvindex")
                precipitation = day.get("precip", 0.0)
                
                # Get date and time
                date_str = day.get("datetime", "")
                data_time = f"{date_str} 12:00:00"  # Use noon as representative time
                
                # Get sunrise and sunset
                sunrise_epoch = day.get("sunriseEpoch", 0)
                sunset_epoch = day.get("sunsetEpoch", 0)
                sunrise = datetime.datetime.fromtimestamp(sunrise_epoch).strftime('%H:%M:%S')
                sunset = datetime.datetime.fromtimestamp(sunset_epoch).strftime('%H:%M:%S')
            else:
                logger.error("No weather data found in the response")
                return None
        
        return WeatherData(
            city_name=address,
            description=description,
            temperature_c=temp,
            feels_like_c=feels_like,
            humidity_pct=humidity,
            pressure_hpa=pressure,
            wind_speed_ms=wind_speed * 0.44704 if not units == "metric" else wind_speed,  # Convert mph to m/s if needed
            wind_direction=deg_to_compass(wind_dir),
            wind_direction_deg=wind_dir,
            cloudiness_pct=cloudiness,
            sunrise=sunrise,
            sunset=sunset,
            data_time=data_time,
            provider=self.get_provider_name(),
            uv_index=uv_index,
            precipitation_mm=precipitation if precipitation > 0 else None,
            historical_date=historical_date.isoformat() if historical_date else None
        )
    
    def _process_forecast_data(self, data: Dict[str, Any], days: int) -> ForecastData:
        """Process forecast data from Visual Crossing"""
        address = data.get("address", "Unknown Location")
        forecast_days_data = data.get("days", [])
        
        # Limit to requested number of days
        forecast_days_data = forecast_days_data[:days]
        
        forecast_days = []
        
        for day_data in forecast_days_data:
            description = day_data.get("conditions", "No description")
            temp = day_data.get("temp", 0.0)
            feels_like = day_data.get("feelslike", 0.0)
            humidity = day_data.get("humidity", 0)
            pressure = day_data.get("pressure", 0)
            wind_speed = day_data.get("windspeed", 0.0)
            wind_dir = day_data.get("winddir", 0)
            cloudiness = day_data.get("cloudcover", 0)
            uv_index = day_data.get("uvindex")
            precipitation = day_data.get("precip", 0.0)
            
            # Get date and time
            date_str = day_data.get("datetime", "")
            data_time = f"{date_str} 12:00:00"  # Use noon as representative time
            
            # Get sunrise and sunset
            sunrise_epoch = day_data.get("sunriseEpoch", 0)
            sunset_epoch = day_data.get("sunsetEpoch", 0)
            sunrise = datetime.datetime.fromtimestamp(sunrise_epoch).strftime('%H:%M:%S')
            sunset = datetime.datetime.fromtimestamp(sunset_epoch).strftime('%H:%M:%S')
            
            forecast_days.append(WeatherData(
                city_name=address,
                description=description,
                temperature_c=temp,
                feels_like_c=feels_like,
                humidity_pct=humidity,
                pressure_hpa=pressure,
                wind_speed_ms=wind_speed * 0.44704 if not units == "metric" else wind_speed,  # Convert mph to m/s if needed
                wind_direction=deg_to_compass(wind_dir),
                wind_direction_deg=wind_dir,
                cloudiness_pct=cloudiness,
                sunrise=sunrise,
                sunset=sunset,
                data_time=data_time,
                provider=self.get_provider_name(),
                uv_index=uv_index,
                precipitation_mm=precipitation if precipitation > 0 else None
            ))
        
        return ForecastData(
            city_name=address,
            forecast_days=forecast_days,
            provider=self.get_provider_name()
        )
    
    def _handle_error_response(self, response: requests.Response) -> None:
        """Handle error responses from the API"""
        logger.error(f"Error {response.status_code}: {response.text}")

class WeatherProviderFactory:
    """Factory for creating weather providers"""
    
    @staticmethod
    def create_provider(provider_name: str, cache: Optional[WeatherCache] = None) -> Optional[WeatherProvider]:
        """Create a provider by name"""
        provider_name = provider_name.lower()
        
        if provider_name == "openweathermap":
            api_key = os.getenv("OPENWEATHER_API_KEY")
            if not api_key:
                logger.error("OpenWeatherMap API key not set")
                return None
            return OpenWeatherMapProvider(api_key, cache)
            
        elif provider_name == "weatherapi":
            api_key = os.getenv("WEATHERAPI_KEY")
            if not api_key:
                logger.error("WeatherAPI key not set")
                return None
            return WeatherAPIProvider(api_key, cache)
            
        elif provider_name == "visualcrossing":
            api_key = os.getenv("VISUALCROSSING_KEY")
            if not api_key:
                logger.error("Visual Crossing API key not set")
                return None
            return VisualCrossingProvider(api_key, cache)
            
        else:
            logger.error(f"Unknown provider: {provider_name}")
            return None

# --------------------------- Utility Functions ------------------------------- #

def deg_to_compass(deg: Optional[int]) -> str:
    """Convert wind direction in degrees to a compass direction."""
    if deg is None:
        return "N/A"
    
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(deg / 22.5) % 16
    return directions[index]

def convert_to_local_time(unix_time: Optional[int], offset: int) -> str:
    """Convert UTC Unix timestamp to local time string using timezone offset."""
    if unix_time is None:
        return "N/A"
    
    utc_time = datetime.datetime.utcfromtimestamp(unix_time)
    local_time = utc_time + datetime.timedelta(seconds=offset)
    return local_time.strftime('%Y-%m-%d %H:%M:%S')

def validate_api_keys() -> bool:
    """Validate that required API keys are set."""
    keys_to_check = [
        ("OPENWEATHER_API_KEY", "OpenWeatherMap")
    ]
    
    # Check if at least one LLM provider key is set
    llm_keys = [
        ("OPENAI_API_KEY", "OpenAI"),
        ("ANTHROPIC_API_KEY", "Anthropic (Claude)")
    ]
    
    all_valid = True
    llm_valid = False
    
    for env_var, service_name in keys_to_check:
        api_key = os.getenv(env_var)
        if not api_key or api_key.startswith("YOUR_"):
            logger.error(f"{service_name} API key is not set. Please set the {env_var} environment variable.")
            all_valid = False
    
    # Check if at least one LLM provider key is set
    for env_var, service_name in llm_keys:
        api_key = os.getenv(env_var)
        if api_key and not api_key.startswith("YOUR_"):
            llm_valid = True
            break
    
    if not llm_valid:
        logger.error("At least one LLM provider API key (OpenAI or Anthropic) must be set.")
        all_valid = False
    
    # Check optional API keys and log warnings
    optional_keys = [
        ("WEATHERAPI_KEY", "WeatherAPI"),
        ("VISUALCROSSING_KEY", "Visual Crossing")
    ]
    
    for env_var, service_name in optional_keys:
        api_key = os.getenv(env_var)
        if not api_key:
            logger.warning(f"{service_name} API key is not set. Some features may be unavailable.")
    
    return all_valid

def get_geographic_context(location: str) -> Optional[GeoContext]:
    """Get geographic context for a location using geopy"""
    if not GEOPY_AVAILABLE:
        logger.warning("geopy is not installed. Geographic context will not be available.")
        return None
        
    try:
        geolocator = geopy.geocoders.Nominatim(user_agent="weather-quill")
        geo_location = geolocator.geocode(location, exactly_one=True, addressdetails=True)
        
        if not geo_location:
            logger.warning(f"Could not get geographic context for {location}")
            return None
            
        address = geo_location.raw.get('address', {})
        
        # Check if location is coastal
        is_coastal = False
        if 'coastline' in geo_location.raw.get('class', '').lower() or 'coastline' in address.get('natural', '').lower():
            is_coastal = True
            
        # Look for nearby water bodies
        water_types = ['sea', 'ocean', 'lake', 'river']
        nearby_water = None
        for water in water_types:
            if water in address or water in geo_location.raw.get('display_name', '').lower():
                nearby_water = water
                break
                
        # Get terrain information if available
        terrain = None
        terrain_types = ['mountain', 'hill', 'valley', 'plain', 'forest', 'desert']
        for t in terrain_types:
            if t in geo_location.raw.get('display_name', '').lower():
                terrain = t
                break
        
        return GeoContext(
            country=address.get('country'),
            region=address.get('state') or address.get('region'),
            is_coastal=is_coastal,
            elevation=geo_location.altitude,
            nearby_water=nearby_water,
            terrain=terrain
        )
        
    except Exception as e:
        logger.warning(f"Error getting geographic context: {e}")
        return None

# --------------------------- Prompt Building --------------------------------- #

def build_current_weather_prompt(weather_data: WeatherData, units: str = "metric") -> str:
    """
    Build a contextual prompt for current weather for the LLM.
    
    Args:
        weather_data: Processed weather data object
        units: Unit system (metric or imperial)
        
    Returns:
        Formatted prompt string for the LLM
    """
    if not weather_data:
        return "No valid weather data found. Please provide correct data."
    
    # Format geographic context if available
    geo_context = ""
    if weather_data.geo_context:
        geo_context = weather_data.geo_context.to_description()
        if geo_context:
            geo_context = f" {geo_context}"
    
    user_prompt = (
        f"As of {weather_data.data_time}, in the city of {weather_data.city_name}{geo_context},\n"
        f"Weather Description: {weather_data.description}\n"
        f"Temperature: {weather_data.temperature_formatted(units)}\n"
        f"Humidity: {weather_data.humidity_pct}%\n"
        f"Pressure: {weather_data.pressure_hpa} hPa\n"
        f"Wind: {weather_data.wind_formatted(units)} ({weather_data.wind_direction})\n"
        f"Cloudiness: {weather_data.cloudiness_pct}%\n"
    )
    
    # Add optional data if available
    if weather_data.sunrise != "N/A":
        user_prompt += f"Sunrise: {weather_data.sunrise}\n"
    if weather_data.sunset != "N/A":
        user_prompt += f"Sunset: {weather_data.sunset}\n"
    if weather_data.uv_index is not None:
        user_prompt += f"UV Index: {weather_data.uv_index}\n"
    if weather_data.precipitation_mm is not None:
        precip_value = weather_data.precipitation_mm
        if units == "imperial":
            precip_value = weather_data.precipitation_mm / 25.4  # Convert to inches
            user_prompt += f"Precipitation: {precip_value:.2f} in\n"
        else:
            user_prompt += f"Precipitation: {precip_value:.2f} mm\n"
    if weather_data.air_quality_index is not None:
        user_prompt += f"Air Quality Index: {weather_data.air_quality_index}\n"
    
    # Add data source
    user_prompt += f"Data Source: {weather_data.provider}\n\n"
    
    # Add instructions
    user_prompt += (
        "Create a theatrical, poetic interpretation of this atmospheric state. "
        "Reference the weather conditions—imagine them as characters on a grand cosmic stage. "
        "Keep it elegant, imaginative, yet grounded in the facts above."
    )
    
    return user_prompt

def build_historical_weather_prompt(weather_data: WeatherData, units: str = "metric") -> str:
    """
    Build a contextual prompt for historical weather for the LLM.
    
    Args:
        weather_data: Processed historical weather data object
        units: Unit system (metric or imperial)
        
    Returns:
        Formatted prompt string for the LLM
    """
    if not weather_data or not weather_data.historical_date:
        return "No valid historical weather data found. Please provide correct data."
    
    # Get the year from the historical date
    try:
        year = datetime.date.fromisoformat(weather_data.historical_date).year
    except:
        year = "an unknown year"
    
    # Format geographic context if available
    geo_context = ""
    if weather_data.geo_context:
        geo_context = weather_data.geo_context.to_description()
        if geo_context:
            geo_context = f" {geo_context}"
    
    user_prompt = (
        f"On {weather_data.historical_date}, in the year {year}, in the city of {weather_data.city_name}{geo_context},\n"
        f"Weather Description: {weather_data.description}\n"
        f"Temperature: {weather_data.temperature_formatted(units)}\n"
        f"Humidity: {weather_data.humidity_pct}%\n"
    )
    
    # Add whatever historical data is available
    if weather_data.pressure_hpa > 0:
        user_prompt += f"Pressure: {weather_data.pressure_hpa} hPa\n"
    if weather_data.wind_speed_ms > 0:
        user_prompt += f"Wind: {weather_data.wind_formatted(units)}\n"
    if weather_data.cloudiness_pct > 0:
        user_prompt += f"Cloudiness: {weather_data.cloudiness_pct}%\n"
    if weather_data.sunrise != "N/A":
        user_prompt += f"Sunrise: {weather_data.sunrise}\n"
    if weather_data.sunset != "N/A":
        user_prompt += f"Sunset: {weather_data.sunset}\n"
    if weather_data.precipitation_mm is not None:
        precip_value = weather_data.precipitation_mm
        if units == "imperial":
            precip_value = weather_data.precipitation_mm / 25.4  # Convert to inches
            user_prompt += f"Precipitation: {precip_value:.2f} in\n"
        else:
            user_prompt += f"Precipitation: {precip_value:.2f} mm\n"
    
    # Add data source
    user_prompt += f"Data Source: {weather_data.provider}\n\n"
    
    # Add instructions for historical weather
    user_prompt += (
        f"Create a theatrical, poetic interpretation of this historical weather from {year}. "
        "Write as if you were a person from that time period observing the weather. "
        "Use period-appropriate language and references if possible. "
        "Consider what historical events might have been happening around this time. "
        "Keep it imaginative yet grounded in the facts above."
    )
    
    return user_prompt

def build_forecast_prompt(forecast_data: ForecastData, units: str = "metric") -> str:
    """
    Build a contextual prompt for forecast weather for the LLM.
    
    Args:
        forecast_data: Processed forecast data object
        units: Unit system (metric or imperial)
        
    Returns:
        Formatted prompt string for the LLM
    """
    if not forecast_data or not forecast_data.forecast_days:
        return "No valid forecast data found. Please provide correct data."
    
    user_prompt = (
        f"Weather forecast for {forecast_data.city_name} for the next {len(forecast_data.forecast_days)} days:\n\n"
    )
    
    # Add each day's forecast
    for i, day in enumerate(forecast_data.forecast_days):
        # Determine if this is today, tomorrow, or a future day
        if i == 0:
            day_label = "Today"
        elif i == 1:
            day_label = "Tomorrow"
        else:
            # Try to parse the date from data_time
            try:
                day_date = datetime.datetime.strptime(day.data_time.split()[0], '%Y-%m-%d')
                day_label = day_date.strftime('%A, %b %d')
            except:
                day_label = f"Day {i+1}"
        
        user_prompt += f"--- {day_label} ---\n"
        user_prompt += f"Weather: {day.description}\n"
        user_prompt += f"Temperature: {day.temperature_formatted(units)}\n"
        user_prompt += f"Humidity: {day.humidity_pct}%\n"
        
        # Add whatever forecast data is available
        if day.wind_speed_ms > 0:
            user_prompt += f"Wind: {day.wind_formatted(units)}\n"
        if day.cloudiness_pct > 0:
            user_prompt += f"Cloudiness: {day.cloudiness_pct}%\n"
        if day.uv_index is not None:
            user_prompt += f"UV Index: {day.uv_index}\n"
        if day.precipitation_mm is not None:
            precip_value = day.precipitation_mm
            if units == "imperial":
                precip_value = day.precipitation_mm / 25.4  # Convert to inches
                user_prompt += f"Precipitation: {precip_value:.2f} in\n"
            else:
                user_prompt += f"Precipitation: {precip_value:.2f} mm\n"
        
        user_prompt += "\n"
    
    # Add data source
    user_prompt += f"Data Source: {forecast_data.provider}\n\n"
    
    # Add instructions for forecast
    user_prompt += (
        "Create a theatrical, poetic forecast narrative that builds suspense and anticipation. "
        "Treat each day as an act in a grand meteorological drama. Use foreshadowing and narrative arcs. "
        "Reference the specific conditions for each day. Make the weather sound like an unfolding story "
        "with a beginning, middle, and end."
    )
    
    return user_prompt

def build_comparative_prompt(weather_data_list: List[WeatherData], units: str = "metric") -> str:
    """
    Build a prompt for comparing weather across multiple locations.
    
    Args:
        weather_data_list: List of WeatherData objects to compare
        units: Unit system (metric or imperial)
        
    Returns:
        Formatted prompt string for the LLM
    """
    if not weather_data_list or len(weather_data_list) < 2:
        return "Need at least two locations to compare. Please provide valid data."
    
    user_prompt = "Compare and contrast the weather conditions in these locations:\n\n"
    
    # Add each location's current weather
    for weather in weather_data_list:
        # Format geographic context if available
        geo_context = ""
        if weather.geo_context:
            geo_context = weather.geo_context.to_description()
            if geo_context:
                geo_context = f" {geo_context}"
        
        user_prompt += f"--- {weather.city_name}{geo_context} ---\n"
        user_prompt += f"Weather: {weather.description}\n"
        user_prompt += f"Temperature: {weather.temperature_formatted(units)}\n"
        user_prompt += f"Humidity: {weather.humidity_pct}%\n"
        user_prompt += f"Wind: {weather.wind_formatted(units)}\n"
        
        # Add optional data if available and relevant for comparison
        if weather.cloudiness_pct > 0:
            user_prompt += f"Cloudiness: {weather.cloudiness_pct}%\n"
        if weather.precipitation_mm is not None:
            precip_value = weather.precipitation_mm
            if units == "imperial":
                precip_value = weather.precipitation_mm / 25.4
                user_prompt += f"Precipitation: {precip_value:.2f} in\n"
            else:
                user_prompt += f"Precipitation: {precip_value:.2f} mm\n"
        
        user_prompt += "\n"
    
    # Add instructions for comparison
    user_prompt += (
        "Create a theatrical narrative that weaves together and compares these different weather stories. "
        "Highlight the contrasts and similarities between the locations. "
        "Imagine the different weathers as characters meeting each other or in dialogue. "
        "What would the heat of one city say to the cold of another? "
        "How might the clouds of one place judge the clear skies of another? "
        "Make it creative, engaging, and grounded in the provided weather data."
    )
    
    return user_prompt

# --------------------------- LLM Interaction --------------------------------- #

def call_llm(system_prompt: str, user_prompt: str, llm_provider: str = "openai",
             model: Optional[str] = None, temperature: float = 0.9, 
             max_tokens: int = 600) -> str:
    """
    Send prompts to the specified LLM provider and return the response.
    
    Args:
        system_prompt: System prompt defining the LLM's role
        user_prompt: User prompt with weather data
        llm_provider: Provider to use ("openai" or "claude")
        model: Model to use (if None, uses provider's default)
        temperature: Creativity parameter (0.0-1.0)
        max_tokens: Maximum tokens in the response
        
    Returns:
        LLM-generated response or error message
    """
    # Create provider instance
    provider = LLMProviderFactory.create_provider(llm_provider, model)
    if not provider:
        return f"Failed to initialize LLM provider: {llm_provider}"
    
    # Generate text
    return provider.generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )

# --------------------------- Output Functions ------------------------------- #

def save_report(report: str, weather_data: Union[WeatherData, ForecastData], 
                output_dir: str = "reports", report_type: str = "current") -> str:
    """
    Save the weather report to a file.
    
    Args:
        report: Generated report text
        weather_data: Weather data used to generate the report
        output_dir: Directory to save reports
        report_type: Type of report (current, historical, forecast, comparison)
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get city name
    if isinstance(weather_data, WeatherData):
        city_name = weather_data.city_name
    else:  # ForecastData
        city_name = weather_data.city_name
    
    # Create a filename with city and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{city_name.replace(' ', '_').lower()}_{report_type}_{timestamp}.txt"
    file_path = output_path / filename
    
    # Write the report to the file
    with open(file_path, 'w') as f:
        f.write(f"WEATHER QUILL REPORT - {report_type.upper()}\n")
        f.write(f"City: {city_name}\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if report_type == "historical" and isinstance(weather_data, WeatherData):
            f.write(f"Historical Date: {weather_data.historical_date}\n")
            
        f.write("\n" + "="*60 + "\n\n")
        f.write(report)
        f.write("\n\n" + "="*60 + "\n")
        f.write("\nRAW WEATHER DATA:\n")
        
        # Write the data based on type
        if isinstance(weather_data, WeatherData):
            for key, value in weather_data.to_dict().items():
                if key != "geo_context":  # Skip complex nested objects
                    f.write(f"{key}: {value}\n")
        else:  # ForecastData
            f.write(f"City: {weather_data.city_name}\n")
            f.write(f"Provider: {weather_data.provider}\n")
            f.write(f"Forecast Days: {len(weather_data.forecast_days)}\n")
    
    logger.info(f"Report saved to {file_path}")
    return str(file_path)

def generate_audio_report(report: str, output_dir: str = "audio", 
                         language: str = "en", output_format: str = "mp3") -> Optional[str]:
    """
    Generate an audio version of the weather report using text-to-speech.
    
    Args:
        report: Text report to convert to speech
        output_dir: Directory to save audio files
        language: Language code for TTS
        output_format: Output audio format (mp3, wav)
        
    Returns:
        Path to the generated audio file or None on failure
    """
    if not GTTS_AVAILABLE:
        logger.warning("gTTS is not installed. Audio generation is not available.")
        return None
    
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create a filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weather_report_{timestamp}.{output_format}"
        file_path = output_path / filename
        
        # Generate speech
        tts = gTTS(text=report, lang=language, slow=False)
        tts.save(str(file_path))
        
        logger.info(f"Audio report saved to {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"Error generating audio report: {e}")
        return None

# --------------------------- Web Interface ---------------------------------- #

def create_flask_app():
    """Create and configure Flask app for Weather-Quill"""
    if not FLASK_AVAILABLE:
        logger.error("Flask is not installed. Web interface is not available.")
        return None
    
    app = Flask(__name__)
    
    # Create cache for API requests
    cache = WeatherCache(ttl=DEFAULT_CONFIG["cache_ttl"])
    
    @app.route('/')
    def index():
        """Render the main page"""
        style_options = list(STYLE_PROMPTS.keys())
        provider_options = ["openweathermap", "weatherapi", "visualcrossing"]
        
        # Add LLM provider options
        llm_providers = []
        
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            llm_providers.append(("openai", "OpenAI GPT"))
            
        # Check for Anthropic API key
        if os.getenv("ANTHROPIC_API_KEY"):
            llm_providers.append(("claude", "Anthropic Claude"))
        
        # If no providers are available, add a placeholder
        if not llm_providers:
            llm_providers.append(("none", "No API Keys Found"))
        
        return render_template('index.html', 
                              styles=style_options,
                              providers=provider_options,
                              llm_providers=llm_providers)
    
    @app.route('/report', methods=['POST'])
    def generate_report():
        """Generate a weather report based on the form data"""
        # Get form data
        city = request.form.get('city', '')
        style = request.form.get('style', 'dramatic')
        provider_name = request.form.get('provider', 'openweathermap')
        report_type = request.form.get('report_type', 'current')
        units = request.form.get('units', 'metric')
        
        # Get LLM provider selection
        llm_provider = request.form.get('llm_provider', DEFAULT_CONFIG["llm_provider"])
        llm_model = request.form.get('llm_model', '')
        
        # Get date for historical reports
        historical_date = None
        if report_type == 'historical':
            date_str = request.form.get('historical_date', '')
            try:
                historical_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return render_template('error.html', 
                                     message="Invalid date format. Please use YYYY-MM-DD.")
        
        # Get number of days for forecast
        forecast_days = 5
        if report_type == 'forecast':
            try:
                forecast_days = int(request.form.get('forecast_days', '5'))
                if forecast_days < 1 or forecast_days > 14:
                    raise ValueError("Days must be between 1 and 14")
            except ValueError:
                return render_template('error.html',
                                     message="Invalid number of forecast days. Please use a number between 1 and 14.")
        
        # Get locations for comparison
        comparison_locations = []
        if report_type == 'comparison':
            locations_str = request.form.get('comparison_locations', '')
            comparison_locations = [loc.strip() for loc in locations_str.split(',') if loc.strip()]
            if len(comparison_locations) < 2:
                return render_template('error.html',
                                     message="Please provide at least two comma-separated locations for comparison.")
        
        # Get the weather provider
        weather_provider = WeatherProviderFactory.create_provider(provider_name, cache)
        if not weather_provider:
            return render_template('error.html',
                                 message=f"Could not initialize weather provider '{provider_name}'. Check your API keys.")
        
        # Generate the report based on type
        try:
            if report_type == 'current':
                # Get current weather
                weather_data = weather_provider.get_current_weather(city, units)
                if not weather_data:
                    return render_template('error.html',
                                         message=f"Could not get weather data for {city}")
                
                # Add geographic context if enabled
                if DEFAULT_CONFIG.get("geolocation_enabled", True):
                    weather_data.geo_context = get_geographic_context(city)
                
                # Build prompt and generate report
                prompt = build_current_weather_prompt(weather_data, units)
                system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["dramatic"])
                
                # Use the selected LLM provider
                report = call_llm(
                    system_prompt, 
                    prompt,
                    llm_provider=llm_provider,
                    model=llm_model if llm_model else None,
                    temperature=DEFAULT_CONFIG["temperature"],
                    max_tokens=DEFAULT_CONFIG["max_tokens"]
                )
                
                return render_template('report.html',
                                     city=city,
                                     report=report,
                                     weather=weather_data.to_dict(),
                                     report_type='current',
                                     style=style,
                                     llm_provider=llm_provider)
                
            elif report_type == 'historical':
                # Get historical weather
                weather_data = weather_provider.get_historical_weather(city, historical_date, units)
                if not weather_data:
                    return render_template('error.html',
                                         message=f"Could not get historical weather data for {city} on {historical_date}")
                
                # Add geographic context if enabled
                if DEFAULT_CONFIG.get("geolocation_enabled", True):
                    weather_data.geo_context = get_geographic_context(city)
                
                # Build prompt and generate report
                prompt = build_historical_weather_prompt(weather_data, units)
                system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["historical"])
                
                # Use the selected LLM provider
                report = call_llm(
                    system_prompt, 
                    prompt,
                    llm_provider=llm_provider,
                    model=llm_model if llm_model else None,
                    temperature=DEFAULT_CONFIG["temperature"],
                    max_tokens=DEFAULT_CONFIG["max_tokens"]
                )
                
                return render_template('report.html',
                                     city=city,
                                     report=report,
                                     weather=weather_data.to_dict(),
                                     report_type='historical',
                                     style=style,
                                     historical_date=historical_date,
                                     llm_provider=llm_provider)
                
            elif report_type == 'forecast':
                # Get forecast
                forecast_data = weather_provider.get_forecast(city, forecast_days, units)
                if not forecast_data:
                    return render_template('error.html',
                                         message=f"Could not get forecast data for {city}")
                
                # Build prompt and generate report
                prompt = build_forecast_prompt(forecast_data, units)
                system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["dramatic"])
                
                # Use the selected LLM provider
                report = call_llm(
                    system_prompt, 
                    prompt,
                    llm_provider=llm_provider,
                    model=llm_model if llm_model else None,
                    temperature=DEFAULT_CONFIG["temperature"],
                    max_tokens=DEFAULT_CONFIG["max_tokens"]
                )
                
                return render_template('report.html',
                                     city=city,
                                     report=report,
                                     forecast=forecast_data.to_dict(),
                                     report_type='forecast',
                                     style=style,
                                     forecast_days=forecast_days,
                                     llm_provider=llm_provider)
                
            elif report_type == 'comparison':
                # Get weather for all locations
                weather_data_list = []
                for location in comparison_locations:
                    weather = weather_provider.get_current_weather(location, units)
                    if weather:
                        # Add geographic context if enabled
                        if DEFAULT_CONFIG.get("geolocation_enabled", True):
                            weather.geo_context = get_geographic_context(location)
                        weather_data_list.append(weather)
                        weather_data_list.append(weather)
                
                if not weather_data_list or len(weather_data_list) < 2:
                    return render_template('error.html',
                                         message=f"Could not get weather data for at least two locations")
                
                # Build prompt and generate comparison report
                prompt = build_comparative_prompt(weather_data_list, units)
                system_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["dramatic"])
                
                # Use the selected LLM provider
                report = call_llm(
                    system_prompt, 
                    prompt,
                    llm_provider=llm_provider,
                    model=llm_model if llm_model else None,
                    temperature=DEFAULT_CONFIG["temperature"],
                    max_tokens=DEFAULT_CONFIG["max_tokens"]
                )
                
                return render_template('report.html',
                                     city="Multiple Locations",
                                     report=report,
                                     locations=[w.to_dict() for w in weather_data_list],
                                     report_type='comparison',
                                     style=style,
                                     llm_provider=llm_provider)
            
            else:
                return render_template('error.html',
                                     message=f"Invalid report type: {report_type}")
                
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return render_template('error.html',
                                 message=f"Error generating report: {str(e)}")
    
    # Add route to get available models for a provider
    @app.route('/llm_models/<provider>', methods=['GET'])
    def get_llm_models(provider):
        """Get available models for a provider"""
        try:
            llm_provider = LLMProviderFactory.create_provider(provider)
            if not llm_provider:
                return jsonify({"error": f"Provider {provider} not available"}), 400
                
            models = llm_provider.get_available_models()
            return jsonify({"models": models})
            
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/audio', methods=['POST'])
    def generate_audio():
        """Generate an audio version of a report"""
        report_text = request.form.get('report_text', '')
        if not report_text:
            return jsonify({"error": "No report text provided"}), 400
            
        try:
            # Generate audio file
            audio_path = generate_audio_report(report_text, output_dir=DEFAULT_CONFIG["audio_dir"])
            
            if not audio_path:
                return jsonify({"error": "Failed to generate audio"}), 500
                
            # Return the audio file
            return send_file(audio_path, as_attachment=True)
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/save_report', methods=['POST'])
    def save_report_endpoint():
        """Save a report to a file and return the file"""
        report_text = request.form.get('report_text', '')
        city = request.form.get('city', 'unknown_location')
        report_type = request.form.get('report_type', 'current')
        
        if not report_text:
            return jsonify({"error": "No report text provided"}), 400
            
        try:
            # Create a simple WeatherData object for saving
            weather_data = WeatherData(
                city_name=city,
                description="",
                temperature_c=0.0,
                feels_like_c=0.0,
                humidity_pct=0,
                pressure_hpa=0,
                wind_speed_ms=0.0,
                wind_direction="",
                wind_direction_deg=0,
                cloudiness_pct=0,
                sunrise="",
                sunset="",
                data_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Save the report
            file_path = save_report(
                report=report_text,
                weather_data=weather_data,
                output_dir=DEFAULT_CONFIG["reports_dir"],
                report_type=report_type
            )
            
            # Return the file path
            return jsonify({"success": True, "file_path": file_path})
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return jsonify({"error": str(e)}), 500
    
    # Create basic templates
    @app.context_processor
    def inject_templates():
        """Inject templates for testing without actual template files"""
        templates = {
            "index.html": """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Weather-Quill - Poetic Weather Reports</title>
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                        h1 { color: #333; }
                        form { margin-top: 20px; }
                        label { display: block; margin-top: 10px; }
                        input, select { margin-top: 5px; padding: 5px; width: 100%; }
                        button { margin-top: 20px; padding: 10px; background: #4CAF50; color: white; border: none; cursor: pointer; }
                        .report-type-options { margin-top: 10px; display: none; }
                        .llm-options { margin-top: 20px; }
                    </style>
                </head>
                <body>
                    <h1>Weather-Quill - Poetic Weather Reports</h1>
                    <form action="/report" method="post">
                        <label for="city">City or Location:</label>
                        <input type="text" id="city" name="city" required>
                        
                        <label for="report_type">Report Type:</label>
                        <select id="report_type" name="report_type" onchange="showOptions()">
                            <option value="current">Current Weather</option>
                            <option value="historical">Historical Weather</option>
                            <option value="forecast">Forecast</option>
                            <option value="comparison">Location Comparison</option>
                        </select>
                        
                        <div id="historical-options" class="report-type-options">
                            <label for="historical_date">Date (YYYY-MM-DD):</label>
                            <input type="date" id="historical_date" name="historical_date">
                        </div>
                        
                        <div id="forecast-options" class="report-type-options">
                            <label for="forecast_days">Days (1-14):</label>
                            <input type="number" id="forecast_days" name="forecast_days" min="1" max="14" value="5">
                        </div>
                        
                        <div id="comparison-options" class="report-type-options">
                            <label for="comparison_locations">Locations (comma-separated):</label>
                            <input type="text" id="comparison_locations" name="comparison_locations" placeholder="New York, London, Tokyo">
                        </div>
                        
                        <label for="style">Style:</label>
                        <select id="style" name="style">
                            {% for style in styles %}
                            <option value="{{ style }}">{{ style|capitalize }}</option>
                            {% endfor %}
                        </select>
                        
                        <label for="provider">Weather Provider:</label>
                        <select id="provider" name="provider">
                            {% for provider in providers %}
                            <option value="{{ provider }}">{{ provider|capitalize }}</option>
                            {% endfor %}
                        </select>
                        
                        <div class="llm-options">
                            <label for="llm_provider">Language Model Provider:</label>
                            <select id="llm_provider" name="llm_provider" onchange="getModelOptions()">
                                {% for provider_id, provider_name in llm_providers %}
                                <option value="{{ provider_id }}">{{ provider_name }}</option>
                                {% endfor %}
                            </select>
                            
                            <label for="llm_model">Model:</label>
                            <select id="llm_model" name="llm_model">
                                <option value="">Default model</option>
                            </select>
                        </div>
                        
                        <label for="units">Units:</label>
                        <select id="units" name="units">
                            <option value="metric">Metric (°C, m/s)</option>
                            <option value="imperial">Imperial (°F, mph)</option>
                        </select>
                        
                        <button type="submit">Generate Report</button>
                    </form>
                    
                    <script>
                        function showOptions() {
                            // Hide all option divs
                            document.querySelectorAll('.report-type-options').forEach(div => {
                                div.style.display = 'none';
                            });
                            
                            // Show the relevant div
                            const reportType = document.getElementById('report_type').value;
                            if (reportType === 'historical') {
                                document.getElementById('historical-options').style.display = 'block';
                            } else if (reportType === 'forecast') {
                                document.getElementById('forecast-options').style.display = 'block';
                            } else if (reportType === 'comparison') {
                                document.getElementById('comparison-options').style.display = 'block';
                            }
                        }
                        
                        function getModelOptions() {
                            const provider = document.getElementById('llm_provider').value;
                            const modelSelect = document.getElementById('llm_model');
                            
                            // Clear existing options
                            modelSelect.innerHTML = '<option value="">Default model</option>';
                            
                            if (provider === 'none') {
                                return;
                            }
                            
                            // Fetch models for this provider
                            fetch('/llm_models/' + provider)
                                .then(response => response.json())
                                .then(data => {
                                    if (data.error) {
                                        console.error(data.error);
                                        return;
                                    }
                                    
                                    data.models.forEach(model => {
                                        const option = document.createElement('option');
                                        option.value = model;
                                        option.textContent = model;
                                        modelSelect.appendChild(option);
                                    });
                                })
                                .catch(error => {
                                    console.error('Error fetching models:', error);
                                });
                        }
                        
                        // Initialize options on page load
                        document.addEventListener('DOMContentLoaded', function() {
                            showOptions();
                            getModelOptions();
                        });
                    </script>
                </body>
                </html>
            """,
            
            "report.html": """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Weather Report - {{ city }}</title>
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                        h1, h2 { color: #333; }
                        pre { white-space: pre-wrap; background: #f7f7f7; padding: 15px; border-radius: 5px; }
                        .report { margin-top: 20px; line-height: 1.6; }
                        .buttons { margin-top: 20px; }
                        button { margin-right: 10px; padding: 10px; background: #4CAF50; color: white; border: none; cursor: pointer; }
                        .weather-data { margin-top: 20px; background: #eee; padding: 15px; border-radius: 5px; }
                    </style>
                </head>
                <body>
                    <h1>Weather Report for {{ city }}</h1>
                    <p>
                        <strong>Style:</strong> {{ style|capitalize }}
                        <strong>Type:</strong> {{ report_type|capitalize }}
                        <strong>Generated with:</strong> {{ llm_provider|capitalize }}
                        {% if report_type == 'historical' and historical_date %}
                            <strong>Date:</strong> {{ historical_date }}
                        {% endif %}
                        {% if report_type == 'forecast' and forecast_days %}
                            <strong>Days:</strong> {{ forecast_days }}
                        {% endif %}
                    </p>
                    
                    <div class="report">
                        <h2>Poetic Weather Interpretation</h2>
                        <pre>{{ report }}</pre>
                    </div>
                    
                    <div class="buttons">
                        <button onclick="generateAudio()">Generate Audio Version</button>
                        <button onclick="saveReport()">Save Report</button>
                        <button onclick="window.location.href='/'">Back to Home</button>
                    </div>
                    
                    <div class="weather-data">
                        <h2>Raw Weather Data</h2>
                        {% if weather %}
                            <pre>{{ weather|tojson(indent=2) }}</pre>
                        {% elif forecast %}
                            <pre>{{ forecast|tojson(indent=2) }}</pre>
                        {% elif locations %}
                            <h3>Locations Compared:</h3>
                            {% for location in locations %}
                                <h4>{{ location.city_name }}</h4>
                                <pre>{{ location|tojson(indent=2) }}</pre>
                            {% endfor %}
                        {% endif %}
                    </div>
                    
                    <script>
                        function generateAudio() {
                            const reportText = document.querySelector('.report pre').textContent;
                            const form = document.createElement('form');
                            form.method = 'POST';
                            form.action = '/audio';
                            
                            const input = document.createElement('input');
                            input.type = 'hidden';
                            input.name = 'report_text';
                            input.value = reportText;
                            
                            form.appendChild(input);
                            document.body.appendChild(form);
                            form.submit();
                        }
                        
                        function saveReport() {
                            const reportText = document.querySelector('.report pre').textContent;
                            const form = document.createElement('form');
                            form.method = 'POST';
                            form.action = '/save_report';
                            
                            const inputReport = document.createElement('input');
                            inputReport.type = 'hidden';
                            inputReport.name = 'report_text';
                            inputReport.value = reportText;
                            
                            const inputCity = document.createElement('input');
                            inputCity.type = 'hidden';
                            inputCity.name = 'city';
                            inputCity.value = '{{ city }}';
                            
                            const inputType = document.createElement('input');
                            inputType.type = 'hidden';
                            inputType.name = 'report_type';
                            inputType.value = '{{ report_type }}';
                            
                            form.appendChild(inputReport);
                            form.appendChild(inputCity);
                            form.appendChild(inputType);
                            document.body.appendChild(form);
                            form.submit();
                        }
                    </script>
                </body>
                </html>
            """,
            
            "error.html": """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Error - Weather-Quill</title>
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                        h1 { color: #d9534f; }
                        .error-box { background: #f2dede; border: 1px solid #ebccd1; padding: 15px; border-radius: 5px; color: #a94442; }
                        button { margin-top: 20px; padding: 10px; background: #5bc0de; color: white; border: none; cursor: pointer; }
                    </style>
                </head>
                <body>
                    <h1>Error</h1>
                    <div class="error-box">
                        <p>{{ message }}</p>
                    </div>
                    <button onclick="window.location.href='/'">Back to Home</button>
                </body>
                </html>
            """
        }
        
        return templates
    
    return app

def run_web_interface(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    """Run the web interface"""
    app = create_flask_app()
    if app:
        logger.info(f"Starting web interface on http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)
    else:
        logger.error("Failed to create Flask app")

# --------------------------- Command Line Interface -------------------------- #

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Weather-Quill: Poetic Weather Reports")
    
    # Create subparsers for different report types
    subparsers = parser.add_subparsers(dest="report_type", help="Type of weather report")
    
    # Current weather parser
    current_parser = subparsers.add_parser("current", help="Get current weather report")
    current_parser.add_argument("city", nargs="?", help="City name to get weather for")
    
    # Historical weather parser
    historical_parser = subparsers.add_parser("historical", help="Get historical weather report")
    historical_parser.add_argument("city", nargs="?", help="City name to get weather for")
    historical_parser.add_argument("--date", "-d", type=str, required=True,
                                help="Historical date (YYYY-MM-DD)")
    
    # Forecast parser
    forecast_parser = subparsers.add_parser("forecast", help="Get weather forecast report")
    forecast_parser.add_argument("city", nargs="?", help="City name to get weather for")
    forecast_parser.add_argument("--days", "-d", type=int, default=5,
                               help="Number of days to forecast (1-14)")
    
    # Comparison parser
    comparison_parser = subparsers.add_parser("comparison", help="Compare weather between locations")
    comparison_parser.add_argument("cities", nargs="*", help="City names to compare")
    
    # Web interface parser
    web_parser = subparsers.add_parser("web", help="Start web interface")
    web_parser.add_argument("--host", default=DEFAULT_CONFIG["web_host"],
                          help="Host to bind the web server to")
    web_parser.add_argument("--port", "-p", type=int, default=DEFAULT_CONFIG["web_port"],
                          help="Port to bind the web server to")
    web_parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    
    # Common arguments for all parsers except web
    for p in [current_parser, historical_parser, forecast_parser, comparison_parser]:
        p.add_argument("--units", "-u", choices=["metric", "imperial"], 
                     default="metric", help="Unit system (metric or imperial)")
        p.add_argument("--style", "-s", choices=list(STYLE_PROMPTS.keys()),
                     default="dramatic", help="Style of the weather report")
        
        # Add LLM provider selection
        p.add_argument("--llm", choices=["openai", "claude"],
                     default=DEFAULT_CONFIG["llm_provider"], 
                     help="Language model provider to use")
        
        # Make the model option more generic for both providers
        p.add_argument("--model", "-m", 
                     help="Specific model to use (provider-dependent)")
        
        p.add_argument("--save", action="store_true",
                     help="Save the report to a file")
        p.add_argument("--audio", "-a", action="store_true",
                     help="Generate audio version of the report")
        p.add_argument("--provider", "-p", choices=["openweathermap", "weatherapi", "visualcrossing"],
                     default="openweathermap", help="Weather provider to use")
    
    # Global arguments
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["reports_dir"],
                      help="Directory to save reports")
    parser.add_argument("--audio-dir", default=DEFAULT_CONFIG["audio_dir"],
                      help="Directory to save audio files")
    parser.add_argument("--no-cache", action="store_true",
                      help="Disable caching of weather data")
    parser.add_argument("--no-geo", action="store_true",
                      help="Disable geographic context enrichment")
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Default to current if no report type specified
    if not args.report_type:
        args.report_type = "current"
        args.city = None
        args.units = "metric"
        args.style = "dramatic"
        args.llm = DEFAULT_CONFIG["llm_provider"]
        args.model = None
        args.save = False
        args.audio = False
        args.provider = "openweathermap"
    
    # Validate arguments
    if args.report_type == "comparison" and len(args.cities) < 2:
        parser.error("At least two cities are required for comparison")
    
    if args.report_type == "historical":
        try:
            datetime.datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            parser.error("Invalid date format. Please use YYYY-MM-DD")
    
    if args.report_type == "forecast" and (args.days < 1 or args.days > 14):
        parser.error("Forecast days must be between 1 and 14")
    
    return args

# --------------------------- Handler Functions ------------------------------ #

def handle_current_report(args, provider) -> int:
    """Handle current weather report"""
    # Get city name from arguments or prompt user
    city = args.city
    if not city:
        city = input("Enter a city name for the current weather report: ")
    
    # Fetch and process weather data
    weather_data = provider.get_current_weather(city, units=args.units)
    if not weather_data:
        logger.error(f"Could not get weather data for {city}")
        return 1
    
    # Add geographic context if enabled
    if not args.no_geo and DEFAULT_CONFIG.get("geolocation_enabled", True):
        weather_data.geo_context = get_geographic_context(city)
    
    # Build prompt and call LLM
    prompt = build_current_weather_prompt(weather_data, units=args.units)
    system_prompt = STYLE_PROMPTS.get(args.style, STYLE_PROMPTS["dramatic"])
    
    # Use the specified LLM provider
    report = call_llm(
        system_prompt, 
        prompt,
        llm_provider=args.llm,
        model=args.model,
        temperature=DEFAULT_CONFIG["temperature"],
        max_tokens=DEFAULT_CONFIG["max_tokens"]
    )
    
    # Output the report
    print("\n====== Theatrical Weather Report ======\n")
    print(report)
    print("\n====== End of Report ======\n")
    
    # Save the report if requested
    if args.save:
        file_path = save_report(
            report=report,
            weather_data=weather_data,
            output_dir=args.output_dir,
            report_type="current"
        )
        print(f"Report saved to: {file_path}")
    
    # Generate audio if requested
    if args.audio:
        if not GTTS_AVAILABLE:
            logger.warning("gTTS is not installed. Audio generation is not available.")
        else:
            audio_path = generate_audio_report(
                report=report,
                output_dir=args.audio_dir
            )
            if audio_path:
                print(f"Audio report saved to: {audio_path}")
    
    return 0

def handle_historical_report(args, provider) -> int:
    """Handle historical weather report"""
    # Get city name from arguments or prompt user
    city = args.city
    if not city:
        city = input("Enter a city name for the historical weather report: ")
    
    # Parse date
    historical_date = datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
    
    # Fetch and process weather data
    weather_data = provider.get_historical_weather(city, historical_date, units=args.units)
    if not weather_data:
        logger.error(f"Could not get historical weather data for {city} on {args.date}")
        return 1
    
    # Add geographic context if enabled
    if not args.no_geo and DEFAULT_CONFIG.get("geolocation_enabled", True):
        weather_data.geo_context = get_geographic_context(city)
    
    # Build prompt and call LLM
    prompt = build_historical_weather_prompt(weather_data, units=args.units)
    system_prompt = STYLE_PROMPTS.get(args.style, STYLE_PROMPTS["historical"])
    
    # Use the specified LLM provider
    report = call_llm(
        system_prompt, 
        prompt,
        llm_provider=args.llm,
        model=args.model,
        temperature=DEFAULT_CONFIG["temperature"],
        max_tokens=DEFAULT_CONFIG["max_tokens"]
    )
    
    # Output the report
    print(f"\n====== Historical Weather Report ({args.date}) ======\n")
    print(report)
    print("\n====== End of Report ======\n")
    
    # Save the report if requested
    if args.save:
        file_path = save_report(
            report=report,
            weather_data=weather_data,
            output_dir=args.output_dir,
            report_type="historical"
        )
        print(f"Report saved to: {file_path}")
    
    # Generate audio if requested
    if args.audio:
        if not GTTS_AVAILABLE:
            logger.warning("gTTS is not installed. Audio generation is not available.")
        else:
            audio_path = generate_audio_report(
                report=report,
                output_dir=args.audio_dir
            )
            if audio_path:
                print(f"Audio report saved to: {audio_path}")
    
    return 0

def handle_forecast_report(args, provider) -> int:
    """Handle forecast weather report"""
    # Get city name from arguments or prompt user
    city = args.city
    if not city:
        city = input("Enter a city name for the weather forecast: ")
    
    # Fetch and process forecast data
    forecast_data = provider.get_forecast(city, days=args.days, units=args.units)
    if not forecast_data:
        logger.error(f"Could not get forecast data for {city}")
        return 1
    
    # Build prompt and call LLM
    prompt = build_forecast_prompt(forecast_data, units=args.units)
    system_prompt = STYLE_PROMPTS.get(args.style, STYLE_PROMPTS["dramatic"])
    
    # Use the specified LLM provider
    report = call_llm(
        system_prompt, 
        prompt,
        llm_provider=args.llm,
        model=args.model,
        temperature=DEFAULT_CONFIG["temperature"],
        max_tokens=DEFAULT_CONFIG["max_tokens"]
    )
    
    # Output the report
    print(f"\n====== Weather Forecast ({args.days} days) ======\n")
    print(report)
    print("\n====== End of Report ======\n")
    
    # Save the report if requested
    if args.save:
        # Create a simple WeatherData for saving
        weather_data = WeatherData(
            city_name=forecast_data.city_name,
            description="Forecast",
            temperature_c=0.0,
            feels_like_c=0.0,
            humidity_pct=0,
            pressure_hpa=0,
            wind_speed_ms=0.0,
            wind_direction="",
            wind_direction_deg=0,
            cloudiness_pct=0,
            sunrise="",
            sunset="",
            data_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            provider=forecast_data.provider
        )
        
        file_path = save_report(
            report=report,
            weather_data=weather_data,
            output_dir=args.output_dir,
            report_type="forecast"
        )
        print(f"Report saved to: {file_path}")
    
    # Generate audio if requested
    if args.audio:
        if not GTTS_AVAILABLE:
            logger.warning("gTTS is not installed. Audio generation is not available.")
        else:
            audio_path = generate_audio_report(
                report=report,
                output_dir=args.audio_dir
            )
            if audio_path:
                print(f"Audio report saved to: {audio_path}")
    
    return 0

def handle_comparison_report(args, provider) -> int:
    """Handle comparison weather report"""
    # Get cities from arguments or prompt user
    cities = args.cities
    if not cities or len(cities) < 2:
        cities_input = input("Enter city names to compare (comma-separated): ")
        cities = [city.strip() for city in cities_input.split(",") if city.strip()]
        if len(cities) < 2:
            logger.error("At least two cities are required for comparison")
            return 1
    
    # Fetch and process weather data for each city
    weather_data_list = []
    for city in cities:
        weather_data = provider.get_current_weather(city, units=args.units)
        if weather_data:
            # Add geographic context if enabled
            if not args.no_geo and DEFAULT_CONFIG.get("geolocation_enabled", True):
                weather_data.geo_context = get_geographic_context(city)
            weather_data_list.append(weather_data)
        else:
            logger.warning(f"Could not get weather data for {city}")
    
    if len(weather_data_list) < 2:
        logger.error("Could not get weather data for at least two cities")
        return 1
    
    # Build prompt and call LLM
    prompt = build_comparative_prompt(weather_data_list, units=args.units)
    system_prompt = STYLE_PROMPTS.get(args.style, STYLE_PROMPTS["dramatic"])
    
    # Use the specified LLM provider
    report = call_llm(
        system_prompt, 
        prompt,
        llm_provider=args.llm,
        model=args.model,
        temperature=DEFAULT_CONFIG["temperature"],
        max_tokens=DEFAULT_CONFIG["max_tokens"]
    )
    
    # Output the report
    city_names = ", ".join(weather.city_name for weather in weather_data_list)
    print(f"\n====== Weather Comparison ({city_names}) ======\n")
    print(report)
    print("\n====== End of Report ======\n")
    
    # Save the report if requested
    if args.save:
        # Create a simple WeatherData for saving
        weather_data = WeatherData(
            city_name="Multiple Cities",
            description="Comparison",
            temperature_c=0.0,
            feels_like_c=0.0,
            humidity_pct=0,
            pressure_hpa=0,
            wind_speed_ms=0.0,
            wind_direction="",
            wind_direction_deg=0,
            cloudiness_pct=0,
            sunrise="",
            sunset="",
            data_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            provider=provider.get_provider_name()
        )
        
        file_path = save_report(
            report=report,
            weather_data=weather_data,
            output_dir=args.output_dir,
            report_type="comparison"
        )
        print(f"Report saved to: {file_path}")
    
    # Generate audio if requested
    if args.audio:
        if not GTTS_AVAILABLE:
            logger.warning("gTTS is not installed. Audio generation is not available.")
        else:
            audio_path = generate_audio_report(
                report=report,
                output_dir=args.audio_dir
            )
            if audio_path:
                print(f"Audio report saved to: {audio_path}")
    
    return 0

# --------------------------- Main Orchestration ------------------------------ #

def main() -> int:
    """
    Orchestrate the Weather-Quill script with improved functionality.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate API keys
    if not validate_api_keys():
        return 1
    
    # Initialize cache if not disabled
    cache = None if args.no_cache else WeatherCache(ttl=DEFAULT_CONFIG["cache_ttl"])
    
    # Start web interface if requested
    if args.report_type == "web":
        run_web_interface(host=args.host, port=args.port, debug=args.debug)
        return 0
    
    # Get weather provider
    provider = WeatherProviderFactory.create_provider(args.provider, cache)
    if not provider:
        logger.error(f"Failed to initialize weather provider: {args.provider}")
        return 1
    
    # Process based on report type
    try:
        if args.report_type == "current":
            return handle_current_report(args, provider)
        elif args.report_type == "historical":
            return handle_historical_report(args, provider)
        elif args.report_type == "forecast":
            return handle_forecast_report(args, provider)
        elif args.report_type == "comparison":
            return handle_comparison_report(args, provider)
        else:
            logger.error(f"Invalid report type: {args.report_type}")
            return 1
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)
                
