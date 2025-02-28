# Weather-Quill

## Transforming Weather Data into Poetic Narratives

Weather-Quill is a Python application that fetches weather data and transforms it into creative, theatrical narratives using AI language models. It turns mundane weather reports into engaging stories by personifying meteorological elements - clouds become characters, winds act as directors, and temperature shifts transform into plot twists.

![Weather-Quill Banner](https://via.placeholder.com/800x200?text=Weather-Quill:+Meteorological+Storytelling)

## 🌟 Features

- **AI-Powered Narrative Generation**:
  - Support for multiple AI models including:
    - OpenAI GPT models (GPT-3.5, GPT-4, etc.)
    - Anthropic Claude models (Claude 3 Opus, Sonnet, Haiku, etc.)
  - Easy switching between models via CLI or web interface
  - Customizable creativity parameters

- **Rich Narrative Styles**: Generate weather reports in multiple literary styles:
  - Dramatic theatrical narratives
  - Shakespearean sonnets and soliloquies
  - Hardboiled noir detective monologues
  - Scientific poetic explanations
  - Melodramatic weather reporting
  - Grim gothic descriptions
  - Child-friendly storytelling
  - Historical period-appropriate writing

- **Comprehensive Weather Data**:
  - Current conditions from multiple providers
  - Historical weather data from specific dates
  - Multi-day weather forecasts
  - Comparison between multiple locations

- **Multiple Weather Providers**:
  - OpenWeatherMap
  - WeatherAPI.com
  - Visual Crossing

- **Enhanced Geographic Context**:
  - Coastal proximity detection
  - Nearby water bodies identification
  - Regional and country context
  - Terrain information

- **Multiple Output Formats**:
  - Text reports (console output)
  - Saved report files
  - Audio narratives (text-to-speech)
  - Web interface with interactive display

## 📋 Prerequisites

- Python 3.7+
- API keys (at least one weather provider and one LLM provider):

  **Weather Providers**:
  - OpenWeatherMap (required): [Get API Key](https://openweathermap.org/api)
  - WeatherAPI.com (optional): [Get API Key](https://www.weatherapi.com/)
  - Visual Crossing (optional): [Get API Key](https://www.visualcrossing.com/)

  **Language Model Providers** (at least one required):
  - OpenAI API key: [Get API Key](https://platform.openai.com/)
  - Anthropic API key: [Get API Key](https://console.anthropic.com/)

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/weather-quill.git
   cd weather-quill
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your API keys as environment variables:
   ```bash
   # Linux/macOS
   export OPENWEATHER_API_KEY="your_openweathermap_key"
   export OPENAI_API_KEY="your_openai_key"  # For GPT models
   export ANTHROPIC_API_KEY="your_anthropic_key"  # For Claude models
   export WEATHERAPI_KEY="your_weatherapi_key"  # Optional
   export VISUALCROSSING_KEY="your_visualcrossing_key"  # Optional
   
   # Windows
   set OPENWEATHER_API_KEY=your_openweathermap_key
   set OPENAI_API_KEY=your_openai_key
   set ANTHROPIC_API_KEY=your_anthropic_key
   set WEATHERAPI_KEY=your_weatherapi_key  # Optional
   set VISUALCROSSING_KEY=your_visualcrossing_key  # Optional
   ```

## 📚 Usage

Weather-Quill has five main modes, each accessible via subcommands:

### Command Line Interface

#### 1. Current Weather Report

```bash
# Using OpenAI GPT
python weather_quill.py current "London" --style dramatic --llm openai

# Using Claude
python weather_quill.py current "London" --style shakespearean --llm claude
```

#### 2. Historical Weather Report

```bash
# Using OpenAI with specific model
python weather_quill.py historical "Paris" --date 2020-07-14 --style historical --llm openai --model gpt-4

# Using Claude with specific model
python weather_quill.py historical "Paris" --date 2020-07-14 --style historical --llm claude --model claude-3-opus-20240229
```

#### 3. Weather Forecast

```bash
# Default settings
python weather_quill.py forecast "Tokyo" --days 5

# With specific model and style
python weather_quill.py forecast "Tokyo" --days 3 --style noir --llm claude
```

#### 4. Location Comparison

```bash
# Compare multiple locations
python weather_quill.py comparison "New York" "London" "Sydney" --style melodramatic --llm openai
```

#### 5. Web Interface

```bash
# Start the web server
python weather_quill.py web --port 5000
```

### Common Options

All report types support these options:

- `--units` / `-u`: Choose `metric` or `imperial` units
- `--style` / `-s`: Select narrative style
- `--provider` / `-p`: Select weather data provider
- `--llm`: Choose language model provider (`openai` or `claude`)
- `--model` / `-m`: Specify a particular model version
- `--save`: Save the report to a file
- `--audio` / `-a`: Generate an audio version
- `--no-cache`: Disable caching of weather data
- `--no-geo`: Disable geographic context enrichment
- `--verbose` / `-v`: Enable detailed logging

### Web Interface

After starting the web server with `python weather_quill.py web`, navigate to `http://localhost:5000` in your browser. The web interface allows you to:

1. Select report type (current, historical, forecast, comparison)
2. Enter locations and parameters
3. Choose narrative style
4. Select LLM provider and model
5. Generate reports that can be:
   - Viewed in the browser
   - Downloaded as text files
   - Converted to audio

## 🎭 Sample Outputs

### Dramatic Style (OpenAI)

```
ACT I: THE MORNING UNVEILED

The stage is set in London, where the atmospheric drama unfolds beneath a canvas of scattered clouds. 
The temperature, a gentle 18°C, plays the protagonist, moving through the city with a flourish that 
feels distinctly warmer at 20°C to those who encounter it. 

Humidity, that ever-present supporting character, lingers at 72% – not quite the star of today's 
performance, but certainly making its presence known in every breath, every scene.

The wind enters stage left, a modest player at 3.5 m/s, carrying whispers from the Southwest. 
It's neither the boisterous villain nor the silent extra, but rather the messenger, delivering 
the script to cloudiness, which occupies just 40% of the grand theater overhead.

[...continues]
```

### Shakespearean Style (Claude)

```
What light through yonder atmosphere breaks?
'Tis Paris, and fair Sol is the radiant dawn!
At fifteen degrees, the air's embrace awakes,
While summer's lease doth hold 'til day is gone.

From southward winds at gentle pace do blow,
Like whispers from a distant lover's sigh;
The clouds, like scattered thoughts, do come and go,
Claiming but half their kingdom in the sky.

[...continues]
```

## 🌐 Language Model Support

Weather-Quill supports two major language model providers, each with different capabilities:

### OpenAI GPT
- Available models: gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o
- Strengths: Wide availability, consistent outputs
- Set up: Requires `OPENAI_API_KEY` environment variable

### Anthropic Claude
- Available models: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307, claude-3.5-sonnet-20240620
- Strengths: Nuanced writing, longer outputs, creative variations
- Set up: Requires `ANTHROPIC_API_KEY` environment variable

You can specify which model to use with the `--llm` flag (openai or claude) and optionally specify a particular model version with `--model`.

## 🔄 Weather API Provider Details

Weather-Quill supports three weather data providers, each with different capabilities and rate limits:

### OpenWeatherMap
- **Free Tier**: 1,000 calls/day, 60 calls/minute
- **Data Available**: Current weather, 5-day forecast, historical (requires subscription)
- **Required For**: Base functionality

### WeatherAPI.com
- **Free Tier**: 1,000,000 calls/month
- **Data Available**: Current weather, 3-day forecast, 7-day history, air quality
- **Benefits**: More detailed data including UV index and air quality

### Visual Crossing
- **Free Tier**: 1,000 records/day
- **Data Available**: Current weather, 15-day forecast, historical (up to 50 years)
- **Benefits**: Extensive historical data

## 💾 Data Caching

Weather-Quill implements a sophisticated caching system to minimize API calls:

- Weather data is cached by default for 10 minutes
- Historical data is cached for longer periods
- Cache can be disabled with the `--no-cache` flag
- Cache is stored in `~/.weather_quill_cache.json`

## 🛠️ Troubleshooting

### Missing API Keys
```
Error: OpenWeatherMap API key is not set. Please set the OPENWEATHER_API_KEY environment variable.
Error: At least one LLM provider API key (OpenAI or Anthropic) must be set.
```
Solution: Ensure you've set the required API keys as environment variables.

### City Not Found
```
Error: City not found. Please check the spelling or try another city.
```
Solution: Verify the city name or try adding the country code (e.g., "London,UK").

### Rate Limit Exceeded
```
Error: API request failed: Rate limit exceeded
```
Solution: Wait until your API rate limit resets or switch to a different provider.

### Language Model Issues
```
Error: Failed to initialize LLM provider: claude
```
Solution: Verify you have the correct API key for the requested model provider.

## 📦 Dependencies

- **requests**: HTTP requests to weather APIs
- **openai**: Interaction with OpenAI's language models
- **anthropic**: Interaction with Claude language models
- **Flask**: Web interface (optional)
- **geopy**: Geographic context enrichment (optional)
- **gTTS**: Text-to-speech for audio reports (optional)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Weather data provided by OpenWeatherMap, WeatherAPI.com, and Visual Crossing
- Text generation powered by OpenAI GPT and Anthropic Claude models
