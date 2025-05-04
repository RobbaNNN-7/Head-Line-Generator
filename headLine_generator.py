import whisper
from pydub import AudioSegment
from pydub.effects import normalize
import google.generativeai as genai
import os 
from dotenv import load_dotenv

load_dotenv()


""" FUNCTION TO PROCESS AUDIO BEFORE TRANSCRIPTION"""
"""
        INPUT: AUDIO FIILE
        OUTPUT: PROCESSED AUDIO FIILE
"""
def process_audio(audio_file):
    if not os.path.exists(audio_file):
        raise FileNotFoundError("File not Found: ",audio_file)
    
    audio = AudioSegment.from_file(audio_file)
    audio = normalize(audio)
    audio = audio.low_pass_filter(1500)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)   

    audio.export("processed_audio.wav",format = "wav")
    
    return "processed_audio.wav"


""" FUNCTION TO TRANSCRIBE AUDIO"""
"""
        INPUT: AUDIO FIILE
        OUTPUT: TRANSCRIPTION OF AUDIO
"""
def transcribe_audio(audio_file):
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: File '{audio_file}' not found")
        return None
        
    model = whisper.load_model("base")
    processed_audio = process_audio(audio_file)

    try:
        result = model.transcribe(processed_audio)
        return result["text"]
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except RuntimeError as e:
        print(f"Error: Runtime error during transcription - {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during transcription: {e}")
        return None
    
""" FUNCTION TO GENERATE HEADLINE FROM TRANSCRIBED TEXT"""
"""
        INPUT: TRANSCRIBED TEXT
        OUTPUT: GENERATED HEADLINE
"""
def generate_headline(text,number_of_headlines = 4):
    # Empty Text
    if not text:
        return None
    

    # Configure API KEY
    genai.configure(api_key = os.getenv("gemini_api_key"))
    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        prompt = f"""You are an experienced news editor at a major news organization. Analyze the following text and generate 4 distinct sets of headlines.

        For each set, create:
        - A primary headline (compelling, concise, under 10 words)
        - A secondary headline/subheading (if needed, providing additional context)

        Give me in exactly the same JSON format as shown below:

        [
            {
                "set_num": "Headline Set Number",
                "primary": "Headline",
                "secondary": "Subheading"
            }
        ]

        
        Guidelines:
        - Use active voice and strong verbs
        - Include key facts, numbers, and quotes when relevant
        - Maintain journalistic standards and accuracy
        - Avoid clickbait or sensationalism
        - Ensure headlines are SEO-friendly
        - Use proper AP style formatting
        - Consider both breaking news and feature story formats
        - Make sure the headline is not too long
        - Make sure the headline is not too short
        - Make sure the headline is not too generic
        - Make sure the headline is not too specific
        - Make sure the headline is not too vague
        - Make sure the headline is not too clickbait
        - Make sure the headline is not too sensational
        - Make sure the headline is not too misleading

        Number of Headlines: {number_of_headlines}
        
        Text to analyze:
        {text}

        Format your response as:
        Set 1:
        Primary: [Headline]
        Secondary: [Subheading if needed]

        Set 2:
        Primary: [Headline]
        Secondary: [Subheading if needed]

        Set 3:
        Primary: [Headline]
        Secondary: [Subheadings if needed]

        Set 4:
        Primary: [Headline]
        Secondary: [Subheadings if needed]

        and more sets according to the number of headlines you are generating.
        """

        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        print("Error Generating Headline: ",e)
        return None
    

print(generate_headline(""));

""" TODO : SENTIMENT ANALYSIS OF HEADLINE"""
"""
        INPUT: HEADLINE
        OUTPUT: SENTIMENT OF THE HEADLINE
"""
def analyze_sentiment(headline):
    return None
   
    









